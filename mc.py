import gym
from numpy.random import choice, rand
from numpy import argmax
from tqdm import tqdm


e = gym.make("Blackjack-v1")
gamma = 1
N = 3000000
eps = 0.6
Q = {}
returns = {}
actions = list(range(e.action_space.n))


# initialize Q table
for sm in range(1, 22):
  for card in range(1, 11):
    for ace in [False, True]:
      for a in actions:
        Q[(sm, card, ace, a)] = 0
        returns[(sm, card, ace, a)] = []


def play(amx=False):
  d = False
  s = e.reset()
  R = 0
  while not d:
    if amx:
      a = argmax([Q[(*s, a)] for a in actions])
    else:
      a = choice(actions)
    s, r, d, _ = e.step(a)
    R += r
  return R


def test(amx=False):
  res = [play(amx) for _ in range(100)]
  wins = 0
  draws = 0
  losses = 0
  for r in res:
    if r == -1: losses += 1
    elif r == 0: draws += 1
    else: wins += 1
  print(f"wins: {wins}\tdraws: {draws}\tlosses: {losses}")


test()

for _ in tqdm(range(N)):
  d = False
  s = e.reset()
  t = 0
  X = []
  R = []
  while not d:
    a = choice(actions) if rand() < eps else argmax([Q[(*s, a)] for a in actions])
    X.append((*s, a))
    s, r, d, _ = e.step(a)
    R.append(r)
    t += 1
  R.append(0)
  for i in range(t - 1, -1, -1):
    R[i] += gamma * R[i + 1]
    L = len(returns[X[i]])
    Q[X[i]] = (L * Q[X[i]] + R[i]) / (L + 1)
    returns[X[i]].append(R[i])

test(True)
