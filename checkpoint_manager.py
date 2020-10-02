import torch
import os
class CheckpointManager():
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def save_checkpoint(self, epoch, iteration,  models, tag):
        cpk_dict = {k: x.state_dict() for k, x in models.items()}
        cpk_dict['epoch'] = epoch
        cpk_dict['iteration'] = iteration
        torch.save(cpk_dict, os.path.join(self.folder_path, '%s-checkpoint-%s.pth.tar' % (tag, str(epoch))))

    @staticmethod    
    def load_checkpoint(checkpoint_path, 
                         image_to_skeleton=None,
                         discriminator=None,
                         skeleton_to_keypoints=None,
                         conditional_generator=None,
                         model_kp_detector=None,
                         optimizer_image_to_skeleton=None,
                         optimizer_discriminator=None,
                         optimizer_skeleton_to_keypoints=None,
                         optimizer_conditional_generator=None,
                         optimizer_kp_detector=None):
        checkpoint = torch.load(checkpoint_path)

        ####### Load Models 
        if model_kp_detector is not None:
            if 'model_kp_detector' in checkpoint.keys():
                model_kp_detector.load_state_dict(checkpoint['model_kp_detector'])
        if image_to_skeleton is not None:
            image_to_skeleton.load_state_dict(checkpoint['image_to_skeleton'])
        if discriminator is not None:
            discriminator.load_state_dict(checkpoint['discriminator'])
        if skeleton_to_keypoints is not None:
            skeleton_to_keypoints.load_state_dict(checkpoint['skeleton_to_keypoints'])
        if conditional_generator is not None:
            conditional_generator.load_state_dict(checkpoint['conditional_generator'])

        ####### Load Optmizer
        if optimizer_image_to_skeleton is not None:
            optimizer_image_to_skeleton.load_state_dict(checkpoint['optimizer_image_to_skeleton']) 
        if optimizer_discriminator is not None:
            optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator']) 
        if optimizer_skeleton_to_keypoints is not None:
            optimizer_skeleton_to_keypoints.load_state_dict(checkpoint['optimizer_skeleton_to_keypoints']) 
        if optimizer_conditional_generator is not None:
            optimizer_conditional_generator.load_state_dict(checkpoint['optimizer_conditional_generator']) 
        if optimizer_kp_detector is not None:
            optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])

        return checkpoint['epoch'], checkpoint['iteration']
