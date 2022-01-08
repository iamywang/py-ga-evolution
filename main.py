import numpy as np
import random
import time

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

pop_list = {}


# 导出图片
def save(pic, path):
    plt.figure(figsize=(5.3, 5.3))
    plt.axis("off")
    plt.tight_layout()
    plt.imsave(path, arr=pic)
    plt.close()


# 适应度函数
def similarity(pic1, pic2):
    score = np.sum(np.square((pic1 - pic2)))
    return score


# 初始化种群
def pops(target, pop):
    x, y, z = target.shape
    for i in range(pop):
        pic = np.random.random((x, y, z))
        score = similarity(pic, target)
        individual = f"g{0}_{i}_n"
        pop_list[individual]['gene'] = pic
        pop_list[individual]['score'] = score


# 交叉变异
def crossover(g, target, pair, mutation):
    pi = pop_list.keys()
    # random
    males = random.sample(pi, int(len(pop_list) / 2))
    females = set(pi) - set(males)
    m = random.sample(males, pair)
    f = random.sample(females, pair)

    # 单点交叉
    nums = min(len(m), len(f))
    for c in range(nums):
        pic1 = pop_list[m[c]]['gene']
        pic2 = pop_list[f[c]]['gene']

        x, y, z = pic1.shape
        pic3 = pic1.copy()

        x1_i = random.sample(range(x), int(x / 2))
        y1_i = random.sample(range(y), int(y / 2))

        x2_i = list(set(range(x)) - set(x1_i))
        y2_i = list(set(range(y)) - set(y1_i))

        temp1 = pic2[x1_i]
        temp1[:, y1_i] = pic1[x1_i][:, y1_i]
        pic3[x1_i] = temp1

        temp2 = pic2[x2_i]
        temp2[:, y2_i] = pic1[x2_i][:, y2_i]
        pic3[x2_i] = temp2

        pic4 = pic3.copy()

        # mutation
        m_x = random.sample(range(x), int(x * mutation))
        m_y = random.sample(range(y), int(x * mutation))

        for i, j in zip(m_x, m_y):
            ccc = random.randint(0, z - 1)
            center_p = pic3[i, j][ccc]
            sx = list(range(max(0, i - 3), min(i + 3, x - 1)))
            sy = list(range(max(0, j - 3), min(j + 3, y - 1)))
            temp = pic4[sx]
            normal_rgba = np.random.normal(center_p, .01, size=temp[:, sy].shape[:2])
            normal_rgba[normal_rgba > 1] = 1
            normal_rgba[normal_rgba < 0] = 0
            temp[:, sy, ccc] = normal_rgba
            pic4[sx] = temp

        # update pop_list
        individual_1 = f"g{g}_{c}_n"
        individual_2 = f"g{g}_{c}_m"
        pop_list[individual_1] = {}
        pop_list[individual_2] = {}

        # new_pic1
        pop_list[individual_1]['gene'] = pic3
        pop_list[individual_1]['score'] = similarity(pic3, target)

        # new_pic2
        pop_list[individual_2]['gene'] = pic4
        pop_list[individual_2]['score'] = similarity(pic4, target)


# 进化
def evolution(count):
    # sort
    scores = [(k, v['score']) for k, v in pop_list.items()]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # delete bad individuals
    for i in range(count):
        del pop_list[scores[i][0]]

    # best individual
    best_individual = pop_list[scores[-1][0]]

    # return result
    return best_individual


# run
def run(count, pic, pop, cross, generations, mutation):
    # set output dir
    out_dir = Path("./" + pic.split('.')[0] + "_" + str(count) + "/")
    out_dir.mkdir(exist_ok=True)

    # import target pic
    target = matplotlib.image.imread(pic)

    # start
    start = time.time()

    # init pop
    pops(target, pop)

    best_value = []

    # iter
    out_count = 100
    for i in range(generations):
        # crossover and mutate
        crossover(i, target, int(cross * pop / 2), mutation)

        # evolution
        best_individual = evolution(int(cross * pop))

        # record best_value
        best_value.append(best_individual['score'])

        # output tmp pics
        if i % out_count == (out_count - 1) or i == 0:
            pic = best_individual['gene']
            path = f"{out_dir}/g{i + 1}_{best_individual['score']:.2f}.png"
            save(pic, path)

    # end
    end = time.time()

    # plt
    plt.plot(best_value)
    plt.title("Best pic")
    plt.ylabel("Similarity")
    plt.xlabel("Times")
    plt.show()

    # save line
    out_dir2 = Path("./lines/")
    out_dir2.mkdir(exist_ok=True)
    plt.savefig("./lines/" + str(count) + ".png")

    # print info
    print("GA-%d:" % count)
    print("time: %fs" % (end - start))
    print("pop: %d, cross: %.2f, mutation: %.2f" % (pop, cross, mutation))
    print("generations: %s/%d, best: %f" % (str(np.argmin(best_value)), generations, best_value[len(best_value) - 1]))


# main
if __name__ == '__main__':
    run(
        count=1,
        pic="mix.png",
        generations=100000,
        pop=10,
        cross=0.6,
        mutation=0.1
    )
    run(
        count=2,
        pic="mix.png",
        generations=100000,
        pop=10,
        cross=0.8,
        mutation=0.1
    )
    run(
        count=3,
        pic="mix.png",
        generations=100000,
        pop=10,
        cross=1,
        mutation=0.1
    )
    run(
        count=4,
        pic="mix.png",
        generations=100000,
        pop=10,
        cross=0.6,
        mutation=0.05
    )
    run(
        count=5,
        pic="mix.png",
        generations=100000,
        pop=10,
        cross=0.6,
        mutation=0.01
    )
    run(
        count=6,
        pic="mix.png",
        generations=100000,
        pop=50,
        cross=0.6,
        mutation=0.1
    )
    run(
        count=7,
        pic="mix.png",
        generations=100000,
        pop=100,
        cross=0.6,
        mutation=0.1
    )
