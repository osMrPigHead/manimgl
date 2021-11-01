from manimlib.constants import *
from manimlib.mobject.types.vectorized_mobject import VMobject
from manimlib.utils.config_ops import digest_config


class ParametricCurve(VMobject):
    '''参数曲线'''
    CONFIG = {
        "t_range": [0, 1, 0.1],
        "epsilon": 1e-8,
        # TODO, automatically figure out discontinuities
        "discontinuities": [],
        "use_smoothing": True,
    }

    def __init__(self, t_func, t_range=None, **kwargs):
        '''
        传入 ``function`` 函数，自变量为参数 ``t`` ，返回值为一个三维点坐标
    
        - ``t_range=[t_mix, t_max, dt]`` : 参数 ``t`` 的取值范围和步进
        - ``discontinuities`` : 间断点列表（在这个列表中的值所对应的点将会是图像的间断点）
        '''
        digest_config(self, kwargs)
        if t_range is not None:
            self.t_range[:len(t_range)] = t_range
        # To be backward compatible with all the scenes specifying t_min, t_max, step_size
        self.t_range = [
            kwargs.get("t_min", self.t_range[0]),
            kwargs.get("t_max", self.t_range[1]),
            kwargs.get("step_size", self.t_range[2]),
        ]
        self.t_func = t_func
        VMobject.__init__(self, **kwargs)

    def get_point_from_function(self, t):
        '''获取 t 值对应的点坐标'''
        return self.t_func(t)

    def init_points(self):
        t_min, t_max, step = self.t_range

        jumps = np.array(self.discontinuities)
        jumps = jumps[(jumps > t_min) & (jumps < t_max)]
        boundary_times = [t_min, t_max, *(jumps - self.epsilon), *(jumps + self.epsilon)]
        boundary_times.sort()
        for t1, t2 in zip(boundary_times[0::2], boundary_times[1::2]):
            t_range = [*np.arange(t1, t2, step), t2]
            points = np.array([self.t_func(t) for t in t_range])
            self.start_new_path(points[0])
            self.add_points_as_corners(points[1:])
        if self.use_smoothing:
            self.make_approximately_smooth()
        return self


class FunctionGraph(ParametricCurve):
    '''y-x 函数图像'''
    CONFIG = {
        "color": YELLOW,
        "x_range": [-8, 8, 0.25],
    }

    def __init__(self, function, x_range=None, **kwargs):
        '''
        传入 ``function`` 函数，自变量为 x ，返回值为 y
    
        - ``x_range=[x_mix, x_max, dx]`` 为自变量 x 的取值范围和步进
        '''
        digest_config(self, kwargs)
        self.function = function

        if x_range is not None:
            self.x_range[:len(x_range)] = x_range

        def parametric_function(t):
            return [t, function(t), 0]

        super().__init__(parametric_function, self.x_range, **kwargs)

    def get_function(self):
        '''返回 y-x 函数'''
        return self.function

    def get_point_from_function(self, x):
        '''给出一个横坐标 x ，返回图像上横坐标为 x 的点坐标'''
        return self.t_func(x)
