# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

#TD0 prediction SARSA

from builtins import range
import numpy as np
import matplotlib.pyplot as plt

class WindyGrid:
   def __init__(self, rows, cols, start):
      self.rows = rows
      self.cols = cols
      self.i = start[0]
      self.j = start[1]

   def set(self, rewards, actions, probs):
      self.rewards = rewards
      self.actions = actions
      self.probs = probs

   def set_state(self, s):
      self.i = s[0]
      self.j = s[1]

   def reset(self):
      self.i = 2
      self.j = 0
      return (self.i, self.j)

   def current_state(self):
      return (self.i, self.j)

   def is_terminal(self, s):
      return s not in self.actions

   def move(self, action):
      s = (self.i, self.j)
      a = action
      next_state_probs = self.probs.get((s, a), {s:1})
      next_states = list(next_state_probs.keys())
      next_probs = list(next_state_probs.values())
      s2_index = np.random.choice(len(next_states), p = next_probs)
      s2 = next_states[s2_index]

      self.i, self.j = s2

      return self.rewards.get(s2, 0)

   def action(self, policy):
      s = (self.i, self.j)
      policy_probs = policy[s]
      next_actions = list(policy_probs.keys())
      next_probs = list(policy_probs.values())
      a = np.random.choice(next_actions, p = next_probs)
      return a

   def undo_move(self, action):
      if action == 'L':
         self.j -= 1
      elif action == 'R':
         self.j += 1
      elif action == 'U':
         self.i -= 1
      elif action == 'D':
         self.i += 1
      assert(self.current_state() in self.all_states())

   def game_over(self):
      return(self.i, self.j) not in self.actions

   def all_states(self):
      return(set(self.actions.keys()) | set(self.rewards.keys()))

def print_values(V, g):
   for i in range(g.rows):
      print("---------------------")
      for j in range(g.cols):
         v = V.get((i,j), 0)
         if v >= 0:
            print(" %.2f|" % v, end = "")
         else:
            print("%.2f|" % v, end = "")
      print("")

def print_policy(P, g):
   for i in range(g.rows):
      print("---------------------")
      for j in range(g.cols):
         a = P.get((i,j), " ")
         print("  %s  |" % a, end = "")
      print("")

def windy_grid(step_cost):
   g = WindyGrid(3, 4, (2, 0))
   rewards = {(0, 3): 1, (1, 3): -1}
   actions = {(0, 0): ('D', 'R'),
      (0, 1): ('L', 'R'),
      (0, 2): ('D', 'L', 'R'),
      (1, 0): ('D', 'U'),
      (1, 2): ('D', 'U', 'R'),
      (2, 0): ('U', 'R'),
      (2, 1): ('L', 'R'),
      (2, 2): ('U', 'L', 'R'),
      (2, 3): ('U', 'L')
   }

   for s in actions.keys():
      rewards[s] = step_cost
   probs = {((0, 0), 'D'): {(1, 0):0.5, (0, 1):0.5},
      ((0, 0), 'R'): {(0, 1):1},
      ((0, 1), 'L'): {(0, 0):0.5, (0, 2):0.5},
      ((0, 1), 'R'): {(0, 2):1},
      ((0, 2), 'D'): {(1, 2):0.5, (0, 3):0.5},
      ((0, 2), 'L'): {(0, 1):0.5, (0, 3):0.5},
      ((0, 2), 'R'): {(0, 3):1},
      ((1, 0), 'D'): {(2, 0):1},
      ((1, 0), 'U'): {(0, 0):1},
      ((1, 2), 'D'): {(2, 2):0.5, (1, 3):0.5},
      ((1, 2), 'U'): {(0, 2):0.5, (1, 3):0.5},
      ((1, 2), 'R'): {(1, 3):1},
      ((1, 2), 'L'): {(1, 2):0.5, (1, 3):0.5},
      ((2, 0), 'U'): {(1, 0):0.5, (2, 1):0.5},
      ((2, 0), 'R'): {(2, 1):1},
      ((2, 1), 'L'): {(2, 0):0.5, (2, 2):0.5},
      ((2, 1), 'R'): {(2, 2):1},
      ((2, 2), 'U'): {(1, 2):0.5, (2, 3):0.5},
      ((2, 2), 'L'): {(2, 1):0.5, (2, 3):0.5},
      ((2, 2), 'R'): {(2, 3):1},
      ((2, 3), 'U'): {(1, 3):1},
      ((2, 3), 'L'): {(2, 2):0.5, (2, 3):0.5}
   }
   g.set(rewards, actions, probs)
   return g

def play_game(grid, policy, max_steps=20):
   start_states = list(grid.actions.keys())
   start_idx = np.random.choice(len(start_states))
   grid.set_state(start_states[start_idx])

   s = grid.current_state()
   a = eps_greedy(policy, s)

   s_list = [s]
   r_list = [0]
   a_list = [a]

   for _ in range(max_steps):
      r = grid.move(a)
      s = grid.current_state()

      r_list.append(r)
      s_list.append(s)

      if grid.game_over():
          break

      a = eps_greedy(policy, s)
      a_list.append(a)

   return s_list, r_list, a_list

def max_dict(d):
   max_val = max(d.values())
   max_keys = [key for key, val in d.items() if val == max_val]
   return np.random.choice(max_keys), max_val

def eps_greedy(Q, s, eps=0.1):
   if np.random.random() < eps:
      return np.random.choice(action_space)
   else:
      return max_dict(Q[s])[0]

small_enough = 1e-3
gamma = 0.9
alpha = 0.1

action_space = ['U', 'D', 'L', 'R']

grid = windy_grid(-0.1)
Q = {}
update_counts = {}
reward_per_episode = []

for s in grid.all_states():
   Q[s] = {}
   for a in action_space:
      Q[s][a] = 0

print("\n")
print("rewards:")
print_values(grid.rewards, grid)

#main cycle
for i in range(5000):
   s = grid.reset()
   a = eps_greedy(Q, s)
   episode_reward = 0

   while not grid.game_over():
      r = grid.move(a)
      s_next = grid.current_state()
      episode_reward += r
      a_next = eps_greedy(Q, s_next)
      Q[s][a] += alpha*(r + gamma*Q[s_next][a_next] - Q[s][a])
      update_counts[s] = update_counts.get(s,0) + 1
      a = a_next
      s = s_next
   reward_per_episode.append(episode_reward)

policy = {}
V = {}

for s in grid.actions.keys():
   a, max_q = max_dict(Q[s])
   policy[s] = a
   V[s] = max_q

print("\n")
print("Values:")
print_values(V, grid)

print("\n")
print("policy:")
print_policy(policy, grid)
print(policy)

plt.plot(reward_per_episode)
plt.title("reward_per_episode")
plt.show()

print("update_count:")
total = np.sum(list(update_counts.values()))
for k, v in update_counts.items():
   update_counts[k] = float(v) / total
print_values(update_counts, grid)
