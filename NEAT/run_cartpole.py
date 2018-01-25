# coding: utf-8

"""
NEAT: NeuroEvolution of Augmenting Topologies

https://github.com/MorvanZhou/Evolutionary-Algorithm/blob/master/tutorial-contents/Using%20Neural%20Nets/NEAT_gym/run_cartpole.py
"""

import numpy as np

import neat
import gym

import visualize

GAME = 'CartPole-v0'
env = gym.make(GAME).unwrapped

CONFIG = "./config"
MAX_EP_STEPS = 300
GENERATION_EP = 10
TRAINING = False
CHECKPOINT = 19

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Q-network.
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        # 各episodeの報酬和．
        ep_r = []

        # 1世代につきGENERATION_EP回のepisodeで評価．
        for ep in range(GENERATION_EP):
            r_sum = 0.
            s = env.reset()

            for t in range(MAX_EP_STEPS):
                q_vals = net.activate(s)
                a = np.argmax(q_vals)
                s_next, r, done, info = env.step(a)
                r_sum += r
                if done:
                    break
                s = s_next

            ep_r.append(r_sum)

        # 適合度を，報酬和が最小だったepisodeでの平均報酬で与える．
        genome.fitness = np.min(ep_r) / float(MAX_EP_STEPS)


def run():
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG)
    pop = neat.population.Population(config)

    # statistics
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    # create checkpoint per 5 generations.
    pop.add_reporter(neat.Checkpointer(5))

    # train for 10 generations
    pop.run(eval_genomes, 20)

    # visualize
    visualize.plot_stats(stats, ylog=False,view=True)
    visualize.plot_species(stats, view=True)


def evaluation():
    # restore population
    p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-%i' % CHECKPOINT)
    # Find the winner in restored population
    winner = p.run(eval_genomes, 1)

    # show winner net
    node_names = {-1: 'In0', -2: 'In1', -3: 'In3', -4: 'In4', 0: 'act1', 1: 'act2'}
    visualize.draw_net(p.config, winner, True, node_names=node_names)

    net = neat.nn.FeedForwardNetwork.create(winner, p.config)
    while True:
        s = env.reset()
        while True:
            env.render()
            a = np.argmax(net.activate(s))
            s, r, done, _ = env.step(a)
            if done:
                break


if __name__ == '__main__':
    if TRAINING:
        run()
    else:
        evaluation()
