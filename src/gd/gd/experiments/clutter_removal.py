import collections
import uuid
import os
import numpy as np
import pandas as pd
import sys
import shutil
from pathlib import Path
from gd.gd import io
from gd.gd.grasp import *
from gd.gd.simulation import ClutterRemovalSim
sys.path.append("./")
from rd.render import blender_init_scene, blender_render, blender_update_sceneobj
from scipy.spatial.transform import Rotation as R
import xml.etree.ElementTree as ET

MAX_CONSECUTIVE_FAILURES = 2

State = collections.namedtuple("State", ["tpc"])

def copydirs(from_file, to_file):
    if not os.path.exists(to_file):   
        os.makedirs(to_file)
    files = os.listdir(from_file)  
    for f in files:
        if os.path.isdir(from_file + '/' + f):  
            copydirs(from_file + '/' + f, to_file + '/' + f)  
        else:
            shutil.copy(from_file + '/' + f, to_file + '/' + f)  



def quaternion_translation_matrix(quat, trans):
    rot = quat.as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rot
    transformation_matrix[:3, 3] = trans

    return transformation_matrix


def create_or_modify_xml(template_path, xml_file, mesh_path, transformation_matrixs, id, extrinsic, args, j):
    if os.path.exists(xml_file):
        modify_xml(xml_file, mesh_path, transformation_matrixs, id, extrinsic, args, j)
        print(f"Modified existing XML file: {xml_file}")
    else:
        create_xml(template_path, xml_file, mesh_path, transformation_matrixs, id, extrinsic, args, j)
        print(f"Created new XML file: {xml_file}")

def modify_xml(xml_file, mesh_path, transformation_matrixs, id, extrinsic, args, j):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    root = modify_tree(root, mesh_path, transformation_matrixs, id, extrinsic, args, j)
    tree = ET.ElementTree(root)
    tree.write(xml_file)

def create_xml(template_path, xml_file, mesh_path, transformation_matrixs, id, extrinsic, args, j):
    tree = ET.parse(template_path)
    root = tree.getroot()
    root = modify_tree(root, mesh_path, transformation_matrixs, id, extrinsic, args, j)
    tree = ET.ElementTree(root)
    tree.write(xml_file)

def modify_tree(root, mesh_path, transformation_matrixs, id, extrinsic, args, j):
    extr = extrinsic[j]
    rf = extr[3]
    camera = np.ones((4,1))
    camera[:3, 0] = extr[0]
    target_c = np.ones((4,1))
    target_c[:3, 0] = extr[1]
    up = extr[2]

    rt_inv = np.linalg.inv(rf)    
    origin = rt_inv@camera
    target = rt_inv@target_c

    origin = ', '.join(format(val, '.4f') for val in origin[:3, 0])
    target = ', '.join(format(-val, '.4f') for val in target[:3, 0])
    up = ', '.join(format(val, '.4f') for val in up)
    
    for shape in root.findall('.//shape'):
            if shape.get('id') == str(id):
                mesh_element = shape.find('./string[@name="filename"]')
                mesh_element.set('value', mesh_path)
                transformation_element = shape.find('./transform/matrix')
                matrix_string = ' '.join(map(str, [item for sublist in transformation_matrixs for item in sublist]))
                transformation_element.set('value', matrix_string)


    sensor_element = root.find(".//sensor")
    if sensor_element is not None:
        lookat_element = sensor_element.find('./transform/lookat')
        lookat_element.set('origin', origin)
        lookat_element.set('target', target)
        lookat_element.set('up', up)
    return root

def rename_file(original_path, new_extension):
    directory, filename = os.path.split(original_path)
    dict_path = os.path.abspath(directory)
    base_name, extension = os.path.splitext(filename)
    new_path = os.path.join(dict_path, base_name+ "_visual" + new_extension)

    return new_path

def run(
    grasp_plan_fn,
    logdir,
    description,
    scene,
    object_set,
    num_objects=5,
    n=6,
    N=None,
    num_rounds=40,
    seed=1,
    sim_gui=True, #modified
    rviz=True, #modified
    round_idx=0,
    renderer_root_dir="",
    gpuid=None,
    args=None,
    render_frame_list=[]
):
    """Run several rounds of simulated clutter removal experiments.

    Each round, m objects are randomly placed in a tray. Then, the grasping pipeline is
    run until (a) no objects remain, (b) the planner failed to find a grasp hypothesis,
    or (c) maximum number of consecutive failed grasp attempts.
    """
    sim = ClutterRemovalSim(scene, object_set, gui=sim_gui, seed=seed, renderer_root_dir=renderer_root_dir, args=args)
    logger = Logger(args.log_root_dir, logdir, description, round_idx)

    # output modality
    output_modality_dict = {'RGB': 1,
                            'IR': 0,
                            'NOCS': 0,
                            'Mask': 1,
                            'Normal': 0}

    for n_round in range(round_idx, round_idx+1):
        urdfs_and_poses_dict = sim.reset(num_objects, round_idx)
            
        renderer, quaternion_list, translation_list, path_scene = blender_init_scene(renderer_root_dir, args.log_root_dir, args.obj_texture_image_root_path, scene, urdfs_and_poses_dict, round_idx, logdir, False, args.material_type, gpuid, output_modality_dict)
    
        render_finished = False
        render_fail_times = 0
        while not render_finished and render_fail_times < 3:
            try:
                blender_render(renderer, quaternion_list, translation_list, path_scene, render_frame_list, output_modality_dict, args.camera_focal, is_init=True)
                render_finished = True
            except:
                render_fail_times += 1
        if not render_finished:
            raise RuntimeError("Blender render failed for 3 times.")

        path_scene_backup = os.path.join(path_scene + "_backup", "%d_init"%n_round)
        if os.path.exists(path_scene_backup) == False:
            os.makedirs(path_scene_backup)
        copydirs(path_scene, path_scene_backup)

        round_id = logger.last_round_id() + 1
        logger.log_round(round_id, sim.num_objects)

        consecutive_failures = 1
        last_label = None

        n_grasp = 0
        while sim.num_objects > 0 and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
            timings = {}

            timings["integration"] = 0

            gt_tsdf, gt_pc, _,extrinsic, look_at = sim.acquire_tsdf(n=n, N=N)
            for i in range(2,len(urdfs_and_poses_dict)+2):
                shape = urdfs_and_poses_dict.get("%d" %(i))
                #c = shape[0]
                quat = R.from_quat([shape[1]])
                # rot = quat.as_euler('xyz', degrees=True)
                trans = shape[2]
                mesh = shape[3]
                new_obj = rename_file(mesh,".obj")
                print (quat)
                print (trans)
                transformation_matrix = quaternion_translation_matrix(quat, trans)
                for j in range (len(extrinsic)):
                    xml_file = f'dict_{n_round}_{j}.xml'
                    folder = "xml"
                    if not os.path.exists(folder):
                        os.makedirs(folder)
                    full_path_xml = os.path.join(folder, xml_file)
                    # create_or_modify_xml("urdf/template.xml", full_path_xml, new_obj, trans, rot, i, extrinsic, args, j)
                    create_or_modify_xml("xml/template.xml", full_path_xml, new_obj, transformation_matrix, i, look_at, args, j)

            if args.method == "graspnerf":
                grasps, scores, timings["planning"] = grasp_plan_fn(render_frame_list, round_idx, n_grasp, gt_tsdf)
            else:
                raise NotImplementedError

            if len(grasps) == 0:
                print("no detections found, abort this round")
                break
            else:
                print(f"{len(grasps)} detections found.")

            # execute grasp
            grasp, score = grasps[0], scores[0]
            (label, _), remain_obj_inws_infos = sim.execute_grasp(grasp, allow_contact=True)

            # render the modified scene after grasping
            obj_name_list = [str(value[0]).split("/")[-1][:-5] for value in remain_obj_inws_infos]
            obj_quat_list = [value[2][[3, 0, 1, 2]] for value in remain_obj_inws_infos]
            obj_trans_list = [value[3] for value in remain_obj_inws_infos]
            obj_uid_list = [value[4] for value in remain_obj_inws_infos]

            # update blender scene
            blender_update_sceneobj(obj_name_list, obj_trans_list, obj_quat_list, obj_uid_list)

            # render updated scene
            render_finished = False
            render_fail_times = 0
            while not render_finished and render_fail_times < 3:
                try:
                    blender_render(renderer, quaternion_list, translation_list, path_scene, render_frame_list, output_modality_dict, args.camera_focal)
                    render_finished = True
                except:
                    render_fail_times += 1
            if not render_finished:
                raise RuntimeError("Blender render failed for 3 times.")
        

            path_scene_backup = os.path.join(path_scene+"_backup", "%d_%d"%(n_round,n_grasp))
            if os.path.exists(path_scene_backup)==False:
                os.makedirs(path_scene_backup)
            copydirs(path_scene, path_scene_backup)

            # log the grasp
            logger.log_grasp(round_id, timings, grasp, score, label)

            if last_label == Label.FAILURE and label == Label.FAILURE:
                consecutive_failures += 1
            else:
                consecutive_failures = 1
            last_label = label

            n_grasp += 1


class Logger(object):
    def __init__(self, log_root_dir, expname, description, round_idx):
        self.logdir = Path(os.path.join(log_root_dir, "exp_results", expname , "%04d"%int(round_idx)))#description
        self.scenes_dir = self.logdir / "scenes"
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.rounds_csv_path = self.logdir / "rounds.csv"
        self.grasps_csv_path = self.logdir / "grasps.csv"
        self._create_csv_files_if_needed()

    def _create_csv_files_if_needed(self):
        if not self.rounds_csv_path.exists():
            io.create_csv(self.rounds_csv_path, ["round_id", "object_count"])

        if not self.grasps_csv_path.exists():
            columns = [
                "round_id",
                "scene_id",
                "qx",
                "qy",
                "qz",
                "qw",
                "x",
                "y",
                "z",
                "width",
                "score",
                "label",
                "integration_time",
                "planning_time",
            ]
            io.create_csv(self.grasps_csv_path, columns)

    def last_round_id(self):
        df = pd.read_csv(self.rounds_csv_path)
        return -1 if df.empty else df["round_id"].max()

    def log_round(self, round_id, object_count):
        io.append_csv(self.rounds_csv_path, round_id, object_count)

    def log_grasp(self, round_id, timings, grasp, score, label):
        # log scene
        scene_id = uuid.uuid4().hex

        # log grasp
        qx, qy, qz, qw = grasp.pose.rotation.as_quat()
        x, y, z = grasp.pose.translation
        width = grasp.width
        label = int(label)
        io.append_csv(
            self.grasps_csv_path,
            round_id,
            scene_id,
            qx,
            qy,
            qz,
            qw,
            x,
            y,
            z,
            width,
            score,
            label,
            timings["integration"],
            timings["planning"],
        )


class Data(object):
    """Object for loading and analyzing experimental data."""

    def __init__(self, logdir):
        self.logdir = logdir
        self.rounds = pd.read_csv(logdir / "rounds.csv")
        self.grasps = pd.read_csv(logdir / "grasps.csv")

    def num_rounds(self):
        return len(self.rounds.index)

    def num_grasps(self):
        return len(self.grasps.index)

    def success_rate(self):
        return self.grasps["label"].mean() * 100

    def percent_cleared(self):
        df = (
            self.grasps[["round_id", "label"]]
            .groupby("round_id")
            .sum()
            .rename(columns={"label": "cleared_count"})
            .merge(self.rounds, on="round_id")
        )
        return df["cleared_count"].sum() / df["object_count"].sum() * 100

    def avg_planning_time(self):
        return self.grasps["planning_time"].mean()

    def read_grasp(self, i):
        scene_id, grasp, label = io.read_grasp(self.grasps, i)
        score = self.grasps.loc[i, "score"]
        scene_data = np.load(self.logdir / "scenes" / (scene_id + ".npz"))

        return scene_data["points"], grasp, score, label