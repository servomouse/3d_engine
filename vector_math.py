import math


def dot(u, v):
    """Dot product of two vectors."""
    return sum(ui * vi for ui, vi in zip(u, v))


def norm(v):
    """Euclidean norm (length) of a vector."""
    return math.sqrt(dot(v, v))


def add(u, v):
    """Vector addition u + v."""
    return [ui + vi for ui, vi in zip(u, v)]


def sub(u, v):
    """Vector subtraction u - v."""
    return [ui - vi for ui, vi in zip(u, v)]


def scale(v, s):
    """Scale vector v by scalar s."""
    return [s * vi for vi in v]


def vector_len(vec):
    s = 0
    for axis in vec:
        s += axis**2
    return math.sqrt(s)


def magnitude(v):
    """Calculate the magnitude of vector v."""
    return math.sqrt(sum(x**2 for x in v))


def normalize(v):
    """Normalize vector v."""
    mag = magnitude(v)
    if mag == 0:
        return [0, 0, 0]  # Return a zero vector if the magnitude is zero
    return [x / mag for x in v]
