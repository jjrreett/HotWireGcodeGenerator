import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Helper Functions


def inch2mm(inch):
    return inch*25.4


def vector2(x, y):
    return np.array((x, y))


def calc_point(x, y):
    return np.array((x, y))


def get_normal(vector, angle):
    # returns a unit vector rotated by angle degrees
    angle = np.deg2rad(angle)
    rot = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    vect_new = np.dot(rot, vector)
    vect_new = vect_new / np.linalg.norm(vect_new)
    return(vect_new)


def line_intersection(line1, line2):
    # takes in iterable of iterables (point1, point2)
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return calc_point(x, y)


class data_linewidth_plot():
    def __init__(self, x, y, **kwargs):
        self.ax = kwargs.pop("ax", plt.gca())
        self.fig = self.ax.get_figure()
        self.lw_data = kwargs.pop("linewidth", 1)
        self.lw = 1
        self.fig.canvas.draw()

        self.ppd = 72./self.fig.dpi
        self.trans = self.ax.transData.transform
        self.linehandle, = self.ax.plot([], [], **kwargs)
        if "label" in kwargs:
            kwargs.pop("label")
        self.line, = self.ax.plot(x, y, **kwargs)
        self.line.set_color(self.linehandle.get_color())
        self._resize()
        self.cid = self.fig.canvas.mpl_connect('draw_event', self._resize)

    def _resize(self, event=None):
        lw = ((self.trans((1, self.lw_data))-self.trans((0, 0)))*self.ppd)[1]
        if lw != self.lw:
            self.line.set_linewidth(lw)
            self.lw = lw
            self._redraw_later()

    def _redraw_later(self):
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.single_shot = True
        self.timer.add_callback(lambda: self.fig.canvas.draw_idle())
        self.timer.start()


class LineDataUnits(Line2D):
    def __init__(self, *args, **kwargs):
        _lw_data = kwargs.pop("linewidth", 1)
        super().__init__(*args, **kwargs)
        self._lw_data = _lw_data

    def _get_lw(self):
        if self.axes is not None:
            ppd = 72./self.axes.figure.dpi
            trans = self.axes.transData.transform
            return ((trans((1, self._lw_data))-trans((0, 0)))*ppd)[1]
        else:
            return 1

    def _set_lw(self, lw):
        self._lw_data = lw

    _linewidth = property(_get_lw, _set_lw)


def linewidth_from_data_units(linewidth, axis, reference='y'):
    """
    Convert a linewidth in data units to linewidth in points.

    Parameters
    ----------
    linewidth: float
        Linewidth in data units of the respective reference-axis
    axis: matplotlib axis
        The axis which is used to extract the relevant transformation
        data (data limits and size must not change afterwards)
    reference: string
        The axis that is taken as a reference for the data width.
        Possible values: 'x' and 'y'. Defaults to 'y'.

    Returns
    -------
    linewidth: float
        Linewidth in points
    """
    fig = axis.get_figure()
    if reference == 'x':
        length = fig.bbox_inches.width * axis.get_position().width
        value_range = np.diff(axis.get_xlim())
    elif reference == 'y':
        length = fig.bbox_inches.height * axis.get_position().height
        value_range = np.diff(axis.get_ylim())
    # Convert length to points
    length *= 72
    # Scale linewidth to value range
    return linewidth * (length / value_range)

# Classes


class Cord:
    def __init__(self, raw_data):
        # Takes in Numpy array of normalized cord points
        self.points = raw_data.copy()
        self.cord_length = 1

    def scale(self, cord_length):
        # Scale the points from origin (leading edge)
        self.points = cord_length * self.points
        self.cord_length = cord_length

    def transform(self, transform_vect):
        # moves the points by the given 2d vector
        self.points = self.points + transform_vect

    def reflect(self):
        # reflect points about y axis
        reflect_matrix = np.array([[-1, 0],
                                   [0, 1]])
        self.points = np.dot(self.points, reflect_matrix)

    def plot(self, ax, color="-b"):
        ax.plot(self.points[:, 0], self.points[:, 1])


class Wing:
    def __init__(self, raw_data, root_cord=1, tip_cord=1, sweep=0, span=100):
        self.root = Cord(raw_data)
        self.tip = Cord(raw_data)

        self.span = span

        self.cords = [self.root, self.tip]

        self.root.scale(root_cord)
        self.tip.scale(tip_cord)

        self.tip.transform(vector2(sweep, 0))

    def plot(self, ax):
        self.root.plot(ax)
        self.tip.plot(ax)

    def trailing_edge_start(self):
        self.root.reflect()
        self.tip.reflect()
        self.root.transform(vector2(self.root.cord_length, 0))
        self.tip.transform(vector2(self.root.cord_length, 0))

    def offset(self, vector):
        for cord in self.cords:
            cord.transform(vector)


class Machine:
    def __init__(self, span, stock):
        # span is distance between two towers
        # stock is list of 2 vectors [array(x_offset, y_offset), array(delta_x, delta_y)]
        self.span = span
        self.stock = stock

        self.left = -span/2
        self.right = span/2

    def plot_stock(self, ax):
        ax.add_patch(Rectangle((self.stock[0][0], self.stock[0][1]),
                               self.stock[1][0], self.stock[1][1],
                               facecolor='pink',
                               fill=True,
                               ))


class Cut:
    def __init__(self, machine, wing, side="left"):
        self.machine = machine
        self.wing = wing

        if side == "right":
            mul = -1
        elif side == "left":
            mul = 1
        self.tip_z = -mul*self.wing.span/2
        self.root_z = mul*self.wing.span/2

    def cutter_comp(self, kerf):
        # apply offset algorithm to cord data
        self.kerf = kerf
        self.cuts = []
        for cord in self.wing.cords:
            self.cuts.append(self.offset_cord(self.kerf, cord.points))

    def offset_cord(self, kerf, data):

        for i, p in enumerate(data):
            if i == 0:
                # first point
                p_next = data[i+1]
                points_outside = kerf/2 * get_normal(p_next-p, 90) + p
            elif i == len(data)-1:
                # last point
                p_last = data[i-1]
                points_outside = np.vstack(
                    [points_outside, kerf/2 * get_normal(p_last-p, -90) + p])
            else:
                p_last = data[i-1]
                points_outside = np.vstack(
                    [points_outside, kerf/2 * get_normal(p_last-p, -90) + p])
                p_next = data[i+1]
                points_outside = np.vstack(
                    [points_outside, kerf/2 * get_normal(p_next-p, 90) + p])

        return points_outside

    def plot(self, ax):
        for cut in self.cuts:
            line = LineDataUnits(cut[:, 0], cut[:, 1],
                                 linewidth=self.kerf, alpha=0.4)
            ax.add_line(line)
            # ax.plot(cut[:, 0], cut[:, 1], linewidth=linewidth_from_data_units(self.kerf, ax))

    def output_points(self):

        print("(Program Start)")
        print("G17 G21 G90 G40 G49 G64")
        print("(Inital Height)")

        for (x, y), (u, z) in zip(self.machine_coords[0], self.machine_coords[1]):

            print(f'G1 X{x:06.2f} Y{y:06.2f} U{u:06.2f} Z{z:06.2f}')

        print("M2")

    def calc_start_end(self):
        # first point and second point make a line
        # root cord
        for i, cut in enumerate(self.cuts):
            point1 = cut[0]
            point2 = cut[1]
            point3 = cut[-2]
            point4 = cut[-1]

            intersection_point = line_intersection((point1, point2), (point3, point4))
            start = calc_point(0, intersection_point[1])
            self.cuts[i] = np.vstack((
                start,
                intersection_point,
                cut,
                intersection_point,
                start,
                calc_point(0, 0)
            ))
            # print(intersection_point)

    def iterate_points(self):
        return zip(self.cuts[0], self.cuts[1])

    def calc_points_in_machine_coords(self):
        self.machine_coords = [np.empty(self.cuts[0].shape), np.empty(self.cuts[0].shape)]
        t_left = (self.machine.left - self.root_z) / (self.tip_z - self.root_z)
        t_right = (self.machine.right - self.root_z) / (self.tip_z - self.root_z)
        for i, ((xroot, yroot), (xtip, ytip)) in enumerate(self.iterate_points()):

            xleft = (xtip - xroot) * t_left + xroot
            yleft = (ytip - yroot) * t_left + yroot
            xright = (xtip - xroot) * t_right + xroot
            yright = (ytip - yroot) * t_right + yroot

            m_cord_left = calc_point(xleft, yleft)
            m_cord_right = calc_point(xright, yright)
            self.machine_coords[0][i, :] = m_cord_left
            self.machine_coords[1][i, :] = m_cord_right
        # print(self.cuts)
        # print(self.machine_coords)

    def plot_mc(self, ax):
        for points in self.machine_coords:
            # print(points)
            ax.plot(points[:, 0], points[:, 1])

            # Main


def main():
    # import data
    data_file = "NACA25012.dat"
    raw_data = np.genfromtxt(data_file)
    # print(raw_data)

    # create cord class

    wing = Wing(raw_data, root_cord=inch2mm(4), tip_cord=inch2mm(2.5), span=inch2mm(20))
    wing.trailing_edge_start()

    stock = (vector2(37, 0), vector2(inch2mm(4)+20, inch2mm(1)))

    machine = Machine(inch2mm(60), stock)

    wing.offset(vector2(51, 8))

    cut = Cut(machine, wing, "left")
    cut.cutter_comp(inch2mm(0.115))

    cut.calc_start_end()
    cut.calc_points_in_machine_coords()

    # plot current
    fig, ax = plt.subplots()
    machine.plot_stock(ax)
    wing.plot(ax)
    cut.plot(ax)
    cut.plot_mc(ax)
    plt.axis("equal")
    plt.show()

    cut.output_points()

    # generate g code


if __name__ == "__main__":
    main()
