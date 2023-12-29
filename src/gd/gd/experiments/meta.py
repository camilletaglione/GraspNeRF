import os
import xml.etree.ElementTree as ET
import numpy as np

def extract_mesh_names(xml_file_path):
    mesh_names = []
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    for shape in root.findall('.//shape'):
        if shape.get('id') == '2' or shape.get('id') == '3' or shape.get('id') == '4'  or shape.get('id') == '5' or shape.get('id') == '6':
            mesh_element = shape.find('./string[@name="filename"]')
            mesh_name = mesh_element.get('value')
            _, mesh = os.path.split(mesh_name)
            mesh_names.append(mesh)

    return mesh_names
            

def extract_camera_look_at(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    sensor_element = root.find(".//sensor")
    if sensor_element is not None:
        look_at_element = sensor_element.find('./transform/lookat')

        if look_at_element is not None:
            tvec = look_at_element.get('origin')
            target = look_at_element.get('target')
            up = look_at_element.get('up')
    
            return tvec, target, up

def extract_id(xml_file_path):
    ids = []
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    for shape in root.findall('.//shape'):
            if shape.get('id') == '2' or shape.get('id') == '3' or shape.get('id') == '4'  or shape.get('id') == '5' or shape.get('id') == '6':
                id = shape.get('id')
                ids.append(id)
            
    return ids

            

def write_mesh_to_txt(output_file, mesh_names, line_number):
    with open(output_file, 'r') as file:
        lines = file.readlines()

    if 1 <= int(line_number) <= len(lines):
        lines[int(line_number)-2] = str(int(line_number) - 1) +' '+ mesh_names + ' 2 5\n'

        with open(output_file, 'w') as file:
            file.writelines(lines)


def write_coordinate_to_txt(origin, output_file, line_number):
    with open(output_file, 'r') as file:
        lines = file.readlines()
        print(len(lines), line_number)

    if 1 <= int(line_number) <= len(lines):
        lines[int(line_number) - 2] = origin.replace(',', '')+'\n'

        with open(output_file, 'w') as file:
            file.writelines(lines)

def prep_mat(vec):
    individual_values = vec.split(', ')
    float_values = [float(value) for value in individual_values]
    my_array = np.array(float_values, dtype=np.float32)
    return my_array
        

def write_camera_matrix(origin, target, up, output_file, line_number):
    # pd.to_numeric(row)
    data = np.load(output_file)
    origin = prep_mat(origin)
    target = prep_mat(target)
    up = prep_mat(up)
    forward = (target - origin) / np.linalg.norm(target - origin)
    right = np.cross(forward, up) / np.linalg.norm(np.cross(forward, up))
    new_up = np.cross(right, forward)
    rotation_matrix = np.column_stack((right, new_up, -forward))
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = -origin
    extrinsic_matrix = np.eye(4)
    extrinsic_matrix[:3, :3] = rotation_matrix
    extrinsic_matrix[:3, 3] = -origin
    print(extrinsic_matrix)
    print(data)
    data[int(line_number),:,:] = extrinsic_matrix
    np.save(output_file,data)
#number.rjust(4, '0')


path = '/home/ctaglione/code/GraspNeRF/xml_rgb'
output_path = '/home/ctaglione/code/GraspNeRF/data/traindata_example/giga_hemisphere_train_demo/pola'

def get_info_xml_from_folder(folder_path, output_folder):
    num = []
    for filename in os.listdir(folder_path):
        
        if filename.endswith(".xml"):
            
            file_number = [caractere for caractere in filename if caractere.isdigit()]
            if len(file_number) > 1:
                numbers = ''.join(file_number[:-1])
                number = ''.join(file_number[-1])


            xml_file_path = os.path.join(folder_path, filename)

            mesh_names = extract_mesh_names(xml_file_path)
            look_at_origin, target, up = extract_camera_look_at(xml_file_path)
            ids = extract_id(xml_file_path)
            #print('********************************')
            output_file_name_mesh = os.path.join(output_folder, numbers, 'meta',number+'.txt')
            output_file_name_cam = os.path.join(output_folder, numbers, 'cam_pos_pc.txt')
            output_file_name_cam_mat = os.path.join(output_folder, numbers, 'camera_pos.npy')
      
            if not os.path.exists(os.path.join(output_folder,numbers,'meta')):
                os.makedirs(os.path.join(output_folder,numbers,'meta'))


            if not(numbers in num):
                tensor_data = np.random.rand(*(int(number)+1, 4, 4))
                np.save(output_file_name_cam_mat, tensor_data)

                with open(output_file_name_cam, 'w') as file:
                    for _ in range(int(number)+1):
                        file.write('\n')
                    print('create a file', numbers,'with', int(number)+1, 'lines')




            with open(output_file_name_mesh, 'w') as file:
                for _ in range(len(ids)):
                    file.write('\n') # TODO:verifier car c'etait mentant
            num.append(numbers)
            
            write_coordinate_to_txt(look_at_origin, output_file_name_cam, number)
            write_camera_matrix(look_at_origin, target, up, output_file_name_cam_mat, number)
            for i in range(len(mesh_names)):
                mesh = mesh_names[i]
                id = ids[i]
                write_mesh_to_txt(output_file_name_mesh, mesh,id)


            

            

get_info_xml_from_folder(path, output_path)


def move_files_by_number(root_folder, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for foldername, subfolders, filenames in os.walk(root_folder):
           
        for filename in filenames:
            file_number = [caractere for caractere in filename if caractere.isdigit()]
            if len(file_number) > 1:
                numbers = ''.join(file_number[:-1])
            if 'depth' in filename:
                subfolder_path = os.path.join(destination_folder, numbers, 'depth')
                if not os.path.exists(subfolder_path):
                    os.makedirs(subfolder_path)
            
            
            else:
                print('wrong file')

            new_filename = file_number[-1]
            destination_path = os.path.join(subfolder_path, new_filename+'.png')
            source_path = os.path.join(foldername, filename)
            print(destination_path)
            print(source_path)
            #shutil.copy(source_path, destination_path)