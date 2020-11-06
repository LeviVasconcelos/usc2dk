#### 
# base implemenation from
# https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#RandomAffine
# Riccardo Franceschini 6-10
# example of basic usage 
# img_trans, aff_matrix = batch_img_affine(data["imgs"])
# kp_rot = batch_kp_affine(data["annots"],aff_matrix)
# ------- to get the inverse
# inverse_aff = inverse_aff_values(aff_matrix)
# inverse_rot = batch_kp_affine(kp_rot,inverse_aff, inverse=True)
#######


import numbers
from collections.abc import Sequence
from typing import Tuple, List, Optional
from torchvision.transforms import functional as F
from torch import Tensor
import math

def _get_image_size(img: Tensor) -> List[int]:
    """Returns image sizea as (w, h)
    """
    if isinstance(img, torch.Tensor):
        return [img.shape[-1], img.shape[-2]] #F_t._get_image_size(img)

    return img.size  #F_pil._get_image_size(img)

class RandomAffine(torch.nn.Module):
    """Random affine transformation of the image keeping center invariant.
    The image can be a PIL Image or a Tensor, in which case it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be applied. Else if shear is a tuple or list of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a tuple or list of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        resample (int, optional): An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to ``PIL.Image.NEAREST``.
            If input is Tensor, only ``PIL.Image.NEAREST`` and ``PIL.Image.BILINEAR`` are supported.
        fillcolor (tuple or int): Optional fill color (Tuple for RGB Image and int for grayscale) for the area
            outside the transform in the output image (Pillow>=5.0.0). This option is not supported for Tensor
            input. Fill value for the area outside the transform in the output image is always 0.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters

    """

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=0, fillcolor=0):
        super().__init__()
        self.degrees = _setup_angle(degrees, name="degrees", req_sizes=(2, ))

        if translate is not None:
            _check_sequence_input(translate, "translate", req_sizes=(2, ))
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            _check_sequence_input(scale, "scale", req_sizes=(2, ))
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            self.shear = _setup_angle(shear, name="shear", req_sizes=(2, 4))
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(
            degrees: List[float],
            translate: Optional[List[float]],
            scale_ranges: Optional[List[float]],
            shears: Optional[List[float]],
            img_size: List[int]
    ) -> Tuple[float, Tuple[int, int], float, Tuple[float, float]]:
        """Get parameters for affine transformation

        Returns:
            params to be passed to the affine transformation
        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        if translate is not None:
            max_dx = float(translate[0] * img_size[0])
            max_dy = float(translate[1] * img_size[1])
            tx = int(round(torch.empty(1).uniform_(-max_dx, max_dx).item()))
            ty = int(round(torch.empty(1).uniform_(-max_dy, max_dy).item()))
            translations = (tx, ty)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = float(torch.empty(1).uniform_(scale_ranges[0], scale_ranges[1]).item())
        else:
            scale = 1.0

        shear_x = shear_y = 0.0
        if shears is not None:
            shear_x = float(torch.empty(1).uniform_(shears[0], shears[1]).item())
            if len(shears) == 4:
                shear_y = float(torch.empty(1).uniform_(shears[2], shears[3]).item())

        shear = (shear_x, shear_y)

        return angle, translations, scale, shear


    def forward(self, img):
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Affine transformed image.
        """

        img_size = _get_image_size(img)

        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img_size)
        return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor), ret


    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)

def inverse_affine(img, ret):
    print(f"direct affine parameters {*ret}")
    ### inverse the ret
    return F.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor), ret 
    
    
def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


def _check_sequence_input(x, name, req_sizes):
    msg = req_sizes[0] if len(req_sizes) < 2 else " or ".join([str(s) for s in req_sizes])
    if not isinstance(x, Sequence):
        raise TypeError("{} should be a sequence of length {}.".format(name, msg))
    if len(x) not in req_sizes:
        raise ValueError("{} should be sequence of length {}.".format(name, msg))


def _setup_angle(x, name, req_sizes=(2, )):
    if isinstance(x, numbers.Number):
        if x < 0:
            raise ValueError("If {} is a single number, it must be positive.".format(name))
        x = [-x, x]
    else:
        _check_sequence_input(x, name, req_sizes)

    return [float(d) for d in x]


def _get_affine_matrix(angle: float, translate: List[float], scale: float, shear: List[float]) -> List[float]:

    #       RSS is rotation with scale and shear matrix
    #       RSS(a, s, (sx, sy)) =
    #       = R(a) * S(s) * SHy(sy) * SHx(sx)
    #       = [ s*cos(a - sy)/cos(sy), s*(-cos(a - sy)*tan(x)/cos(y) - sin(a)), 0 ]
    #         [ s*sin(a + sy)/cos(sy), s*(-sin(a - sy)*tan(x)/cos(y) + cos(a)), 0 ]
    #         [ 0                    , 0                                      , 1 ]
    #

    rot = math.radians(angle)
    sx, sy = [math.radians(s) for s in shear]

    # RSS without scaling
    a = math.cos(rot - sy) / math.cos(sy)
    b = -math.cos(rot - sy) * math.tan(sx) / math.cos(sy) - math.sin(rot)
    c = math.sin(rot - sy) / math.cos(sy)
    d = -math.sin(rot - sy) * math.tan(sx) / math.cos(sy) + math.cos(rot)


    matrix = [[a, b, 0], 
          [c, d, 0],
          [0.0,0.0,1.0]]
    matrix = np.array(matrix)
    matrix = [x * scale for x in matrix]
    matrix = torch.Tensor(matrix)


    return matrix

def batch_kp_affine(kps, matrices, inverse=False):
    trans_kps = torch.Tensor([])
    for i in range(len(matrices)):
        # get the rss matrix 
        matrix  = _get_affine_matrix(*matrices[i])
        rot_kps = kps[i] 
        # use the original matrices because it has the right original translation values
        tx = matrices[i][1][0]
        ty = matrices[i][1][1]
        # concatenate for matrix multipliation
        if inverse:
            rot_kps = unnorm_kp(rot_kps)
            rot_kps[:, 0]+= tx
            rot_kps[:, 1]+= ty
            rot_kps = norm_kp(rot_kps)

            
        ones = torch.zeros(15).unsqueeze(1) +1
        rot_kps = torch.cat((rot_kps, ones ),dim=-1)
        rot_kps = torch.matmul(rot_kps, matrix.t()) 

        if not inverse:
            rot_kps = unnorm_kp(rot_kps)
            # translate the unnormalize because the translation values is over the real image
            # so it has to use the unnorm kp then go back to normalized values
            rot_kps[:, 0]+= tx
            rot_kps[:, 1]+= ty
            rot_kps = norm_kp(rot_kps)
        
        trans_kps = torch.cat((trans_kps, rot_kps[:,:-1].unsqueeze(0)), 0)
        #trans_kps.append(rot_kps)
    
    return trans_kps


def batch_img_affine(imgs, degrees_r = [0,360] ,translate_r= (0.1, 0.1), scale_r= (0.4,1.0),shear_r= (0,0)):
     
    warp = RandomAffine(
        degrees=degrees_r,
        translate=translate_r,
        scale=scale_r,
        shear=shear_r,
        resample=Image.BICUBIC) 
        
    transforms = torchvision.transforms.Compose([warp])
    img_trans, aff_matrices = torch.Tensor([]), []

    for mg in imgs:
        tm_img, tm_aff_matrix = transforms(Image.fromarray(np.rollaxis(tensor_to_image(mg), 0, 3)))
        tm_img = torchvision.transforms.ToTensor()(tm_img).unsqueeze(0)
        
        img_trans = torch.cat((img_trans,tm_img), 0)
        aff_matrices.append(tm_aff_matrix)

    return img_trans, aff_matrices


def inverse_aff_values(aff_matrices):
    inverse_aff = [] 
    for matrix in aff_matrices:
        inverse_aff.append( ( -1*matrix[0], (-1*matrix[1][0],-1*matrix[1][1]), 1/matrix[2], (matrix[3][0],matrix[3][1]) ) )
    return inverse_aff

def norm_kp(kps):
    return (2./127.) * (kps) - 1 