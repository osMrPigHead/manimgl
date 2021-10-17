from manimlib.constants import *
from manimlib.mobject.mobject import Mobject
from manimlib.utils.color import color_gradient
from manimlib.utils.color import color_to_rgba
from manimlib.utils.iterables import resize_with_interpolation
from manimlib.utils.iterables import resize_array


class PMobject(Mobject):
    '''点物件'''
    CONFIG = {
        "opacity": 1.0,
    }

    def resize_points(self, size, resize_func=resize_array):
        # TODO
        for key in self.data:
            if key == "bounding_box":
                continue
            if len(self.data[key]) != size:
                self.data[key] = resize_array(self.data[key], size)
        return self

    def set_points(self, points):
        super().set_points(points)
        self.resize_points(len(points))
        return self

    def add_points(self, points, rgbas=None, color=None, opacity=None):
        """
        添加点，点必须是若干个三维坐标，即 Nx3 的数组
        """
        self.append_points(points)
        # rgbas array will have been resized with points
        if color is not None:
            if opacity is None:
                opacity = self.data["rgbas"][-1, 3]
            new_rgbas = np.repeat(
                [color_to_rgba(color, opacity)],
                len(points),
                axis=0
            )
        elif rgbas is not None:
            new_rgbas = rgbas
        self.data["rgbas"][-len(new_rgbas):] = new_rgbas
        return self

    def set_color_by_gradient(self, *colors):
        self.data["rgbas"] = np.array(list(map(
            color_to_rgba,
            color_gradient(colors, self.get_num_points())
        )))
        return self

    def match_colors(self, pmobject):
        self.data["rgbas"][:] = resize_with_interpolation(
            pmobject.data["rgbas"], self.get_num_points()
        )
        return self

    def filter_out(self, condition):
        for mob in self.family_members_with_points():
            to_keep = ~np.apply_along_axis(condition, 1, mob.get_points())
            for key in mob.data:
                if key == "bounding_box":
                    continue
                mob.data[key] = mob.data[key][to_keep]
        return self

    def sort_points(self, function=lambda p: p[0]):
        """
        按照传入的函数对点进行排序，函数接受一个 **三维** 坐标，返回一个数值
        """
        for mob in self.family_members_with_points():
            indices = np.argsort(
                np.apply_along_axis(function, 1, mob.get_points())
            )
            for key in mob.data:
                mob.data[key] = mob.data[key][indices]
        return self

    def ingest_submobjects(self):
        for key in self.data:
            self.data[key] = np.vstack([
                sm.data[key]
                for sm in self.get_family()
            ])
        return self

    def point_from_proportion(self, alpha):
        '''获取点集上百分比为 alpha 的最接近的点'''
        index = alpha * (self.get_num_points() - 1)
        return self.get_points()[int(index)]

    def pointwise_become_partial(self, pmobject, a, b):
        '''获取点集上百分比从 a 到 b 的点'''
        lower_index = int(a * pmobject.get_num_points())
        upper_index = int(b * pmobject.get_num_points())
        for key in self.data:
            if key == "bounding_box":
                continue
            self.data[key] = pmobject.data[key][lower_index:upper_index].copy()
        return self


class PGroup(PMobject):
    '''点集组合'''
    def __init__(self, *pmobs, **kwargs):
        if not all([isinstance(m, PMobject) for m in pmobs]):
            raise Exception("All submobjects must be of type PMobject")
        super().__init__(*pmobs, **kwargs)


class Point(PMobject):
    '''单个点（似乎和 Mobject 中的 Point 类冲突了）'''
    CONFIG = {
        "color": BLACK,
    }

    def __init__(self, location=ORIGIN, **kwargs):
        super().__init__(**kwargs)
        self.add_points([location])
