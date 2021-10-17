import numpy as np
import moderngl

from manimlib.constants import GREY_C
from manimlib.constants import ORIGIN
from manimlib.mobject.types.point_cloud_mobject import PMobject
from manimlib.utils.iterables import resize_preserving_order


DEFAULT_DOT_RADIUS = 0.05
DEFAULT_GRID_HEIGHT = 6
DEFAULT_BUFF_RATIO = 0.5


class DotCloud(PMobject):
    '''点云图'''
    CONFIG = {
        "color": GREY_C,
        "opacity": 1,
        "radius": DEFAULT_DOT_RADIUS,
        "shader_folder": "true_dot",
        "render_primitive": moderngl.POINTS,
        "shader_dtype": [
            ('point', np.float32, (3,)),
            ('radius', np.float32, (1,)),
            ('color', np.float32, (4,)),
        ],
    }

    def __init__(self, points=None, **kwargs):
        '''传入一系列三维坐标，在这些坐标的位置生成点物件'''
        super().__init__(**kwargs)
        if points is not None:
            self.set_points(points)

    def init_data(self):
        super().init_data()
        self.data["radii"] = np.zeros((1, 1))
        self.set_radius(self.radius)

    def to_grid(self, n_rows, n_cols, n_layers=1,
                buff_ratio=None,
                h_buff_ratio=1.0,
                v_buff_ratio=1.0,
                d_buff_ratio=1.0,
                height=DEFAULT_GRID_HEIGHT,
                ):
        '''重置点的数量为 ``n_rows*n_cols*n_layers``，并将点按照 [行, 列, 层] 排列'''
        n_points = n_rows * n_cols * n_layers
        points = np.repeat(range(n_points), 3, axis=0).reshape((n_points, 3))
        points[:, 0] = points[:, 0] % n_cols
        points[:, 1] = (points[:, 1] // n_cols) % n_rows
        points[:, 2] = points[:, 2] // (n_rows * n_cols)
        self.set_points(points.astype(float))

        if buff_ratio is not None:
            v_buff_ratio = buff_ratio
            h_buff_ratio = buff_ratio
            d_buff_ratio = buff_ratio

        radius = self.get_radius()
        ns = [n_cols, n_rows, n_layers]
        brs = [h_buff_ratio, v_buff_ratio, d_buff_ratio]
        self.set_radius(0)
        for n, br, dim in zip(ns, brs, range(3)):
            self.rescale_to_fit(2 * radius * (1 + br) * (n - 1), dim, stretch=True)
        self.set_radius(radius)
        if height is not None:
            self.set_height(height)
        self.center()
        return self

    def set_radii(self, radii):
        '''传入一个数组，逐一设置点的半径'''
        n_points = len(self.get_points())
        radii = np.array(radii).reshape((len(radii), 1))
        self.data["radii"] = resize_preserving_order(radii, n_points)
        self.refresh_bounding_box()
        return self

    def get_radii(self):
        '''获取所有点的半径'''
        return self.data["radii"]

    def set_radius(self, radius):
        '''传入一个数值，统一设置点的半径'''
        self.data["radii"][:] = radius
        self.refresh_bounding_box()
        return self

    def get_radius(self):
        '''获取点半径的最大值'''
        return self.get_radii().max()

    def compute_bounding_box(self):
        bb = super().compute_bounding_box()
        radius = self.get_radius()
        bb[0] += np.full((3,), -radius)
        bb[2] += np.full((3,), radius)
        return bb

    def scale(self, scale_factor, scale_radii=True, **kwargs):
        '''点集大小，``scale_radii`` 控制是否同时缩放每个点的大小'''
        super().scale(scale_factor, **kwargs)
        if scale_radii:
            self.set_radii(scale_factor * self.get_radii())
        return self

    def make_3d(self, gloss=0.5, shadow=0.2):
        '''给点集添加光泽'''
        self.set_gloss(gloss)
        self.set_shadow(shadow)
        self.apply_depth_test()
        return self

    def get_shader_data(self):
        shader_data = super().get_shader_data()
        self.read_data_to_shader(shader_data, "radius", "radii")
        self.read_data_to_shader(shader_data, "color", "rgbas")
        return shader_data


class TrueDot(DotCloud):
    '''一个单点'''
    def __init__(self, center=ORIGIN, radius=DEFAULT_DOT_RADIUS, **kwargs):
        '''
        - ``center`` : 点的中心
        - ``radius`` : 点的半径
        '''
        super().__init__(points=[center], radius=radius, **kwargs)
