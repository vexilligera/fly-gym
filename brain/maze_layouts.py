"""Fixed maze geometry. Topology statistics are for evaluation/display only."""
from dataclasses import dataclass


@dataclass(frozen=True)
class MazeLayout:
    key: str
    name: str
    walls: tuple
    start: tuple
    description: str
    cells: int = 0
    dead_ends: int = 0
    route_turns: int = 0


BOUNDARY = ((-20,0,.5,20.5),(20,0,.5,20.5),(0,-20,20,.5),(0,20,20,.5))
# A fixed 5x5 spanning-tree maze, cell index x + 5*y. Every cell is reachable.
# Keep the maze identical between sensory comparisons and leg-phase seeds.
PASSAGES = ((0,5),(5,6),(6,11),(11,16),(16,17),(17,18),(18,13),(13,14),
            (14,9),(9,8),(8,7),(7,12),(7,2),(2,1),(2,3),(3,4),(14,19),
            (19,24),(24,23),(23,22),(22,21),(21,20),(20,15),(15,10))


def complex_walls():
    openings = {frozenset(pair) for pair in PASSAGES}
    walls = list(BOUNDARY)
    for y in range(5):
        for x in range(5):
            cell = x + 5*y
            if x < 4 and frozenset((cell,cell+1)) not in openings:
                walls.append((-12+8*x,-16+8*y,.5,4.5))
            if y < 4 and frozenset((cell,cell+5)) not in openings:
                walls.append((-16+8*x,-12+8*y,4.5,.5))
    return tuple(walls)


LAYOUTS = {
    'simple': MazeLayout('simple','Simple maze',BOUNDARY+((-6,-8.5,.5,11.5),(6,8.5,.5,11.5)),
                         (-13,-12),'Two offset baffles and wide corridors.'),
    'complex': MazeLayout('complex','Branching maze',complex_walls(),(-16,-16),
                          '25 cells, 5 dead ends, and 8 turns on the route to the central sugar.',25,5,8),
}


def get_layout(key):
    if not isinstance(key,str) or key not in LAYOUTS:
        raise ValueError('Maze layout must be simple or complex')
    return LAYOUTS[key]
