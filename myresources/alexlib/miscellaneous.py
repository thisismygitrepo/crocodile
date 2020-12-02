
import alexlib.toolbox as tb


def compute_num_of_lines_of_code_in_repo(path=tb.P.cwd(), extension=".py", r=True, **kwargs):
    return tb.P(path).myglob(f"*{extension}", r=r, **kwargs).read_text().splitlines().apply(len).np.sum()


def polygon_area(points):
    """Return the area of the polygon whose vertices are given by the
    sequence points.
    """
    area = 0
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p
    # 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    return abs(area / 2)

