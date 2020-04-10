# Glenn Thomas
# 2020-04-09

import grid_world as gw


print(gw.standard_grid().rewards)

g = gw.negative_grid()
print(g.rewards)
print(g.is_terminal((0, 3)))
print(g.game_over())
print(g.all_states())