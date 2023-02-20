import numpy as np
import SimpleITK as sitk

def get_random_translation_3D(distance, N):
    if distance == 0:
        return [0], [0], [0]
    distance = float(distance)
    x = np.random.uniform(-2, 2, size=N)
    y_max = np.sqrt(distance**2-x**2)
    y = np.random.uniform(-y_max, y_max, size=N)
    z = np.sqrt(distance**2-x**2-y**2)*np.random.choice([-1, 1], size=N)
    
    xx = np.array([x, y, z]).T
    
    np.apply_along_axis(np.random.shuffle, 1, xx)
    
    x = xx[:,0]
    y = xx[:,1]
    z = xx[:,2]
    return x, y, z


def transform_and_get_patch(distance, N, mr_sitk, resample, roi_filter):
    for x, y, z in zip(*get_random_translation_3D(distance, N)):
        sitk_transform = sitk.Euler3DTransform()
        # sitk_transform.SetCenter((c1, c2, c3))
        # sitk_transform.SetRotation(rot1, rot2, rot3)
        translation = (x, y, z)
        sitk_transform.SetTranslation(translation)
        resample.SetTransform(sitk_transform)
        output_mr_sitk = resample.Execute(mr_sitk)
        output_mr_sitk = roi_filter.Execute(output_mr_sitk)
        # sitk.WriteImage(output_mr_sitk, '/media/medical/projects/head_and_neck/nnUnet/Task208_ONKOI-bothM-curatedFinal-MR-denoise/translated_test.nii.gz')
        
        yield sitk.GetArrayFromImage(output_mr_sitk), translation
        
        
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

def plot_grad_flow(named_parameters, epoch):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and ('seg_outputs' not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    ave_grads = [i.cpu().item() for i in ave_grads]
    max_grads = [i.cpu().item() for i in max_grads]
    layers, max_grads, ave_grads = map(list, zip(*sorted(zip(layers, max_grads, ave_grads), reverse=False)))
    plt.figure(figsize=(30, 10))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.savefig(f'/media/medical/projects/head_and_neck/nnUnet/Task207_ONKOI-bothM-curatedFinal/gradients_epoch{epoch}.png')
    
    
def register_patches(fixed_patch, moving_patch, rigid=True, deformable=False, root_dir=None):
    rigid_config_fp="/media/medical/gasperp/projects/onkoi_elastix_and_hn_data_preprocessing/resources/SimpleElastixConfig/SimpleElastix_rigid2_patch.txt" 
    deformable_config_fp="/media/medical/gasperp/projects/onkoi_elastix_and_hn_data_preprocessing/resources/SimpleElastixConfig/Parameters0023_Deformable_patch.txt"
    
    if root_dir is None:
        root_dir = '/media/medical/projects/head_and_neck/nnUnet/Task208_ONKOI-bothM-curatedFinal-MR-denoise'
    
    configs = []
    if rigid:
        configs.append(rigid_config_fp)
    if deformable:
        configs.append(deformable_config_fp)
    
    mr_rough_patch_min_intensity = sitk.GetArrayFromImage(moving_patch).min()
    
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.LogToConsoleOn()
    elastixImageFilter.SetOutputDirectory(root_dir)
    elastixImageFilter.LogToFileOn()
    elastixImageFilter.SetFixedImage(fixed_patch)
    elastixImageFilter.SetMovingImage(moving_patch)

    for enum, config_fp in enumerate(configs):
        parameter_map = sitk.ReadParameterFile(config_fp)
        parameter_map['DefaultPixelValue'] = (str(mr_rough_patch_min_intensity),)
        if enum == 0:
            elastixImageFilter.SetParameterMap(parameter_map)
        else:
            elastixImageFilter.AddParameterMap(parameter_map)
            
    # execute registration
    elastixImageFilter.Execute()
    # result_image = elastixImageFilter.GetResultImage()

    transformParameterMapList = elastixImageFilter.GetTransformParameterMap()
    return transformParameterMapList


def apply_transform(moving_image, transformParameterMapList):
    
    for pmap in transformParameterMapList:
        pmap['Origin'] = [str(i) for i in moving_image.GetOrigin()]
        pmap['Size'] = [str(i) for i in moving_image.GetSize()]
    
    transformixImageFilter = sitk.TransformixImageFilter()
    # setup so that elastix writes to main log file
    transformixImageFilter.SetLogToConsole(False)
    transformixImageFilter.SetLogToFile(True)
    transformixImageFilter.SetTransformParameterMap(transformParameterMapList)
    transformixImageFilter.SetMovingImage(moving_image)
    transformixImageFilter.Execute()
    transformed_moving_image = transformixImageFilter.GetResultImage()
    return transformed_moving_image