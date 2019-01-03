from collections import deque
import random


class Memory(object):
    def __init__(self):
        self.container_good = deque()
        self.container_normal = deque()
        self.container_bad = deque()

    def add(self, weight, memory):
        if weight > 0.1:
            self.container_good.append(memory)
        elif weight > 0:
            self.container_normal.append(memory)
        else:
            self.container_bad.append(memory)

    def sample(self, num):
        num = int(num / 3)
        good = random.sample(self.container_good, num) if len(
            self.container_good) > num else self.container_good
        normal = random.sample(self.container_normal, num) if len(
            self.container_normal) > num else self.container_normal
        bad = random.sample(self.container_bad, num) if len(
            self.container_bad) > num else self.container_bad
        result = []
        result += good
        result += normal
        result += bad
        # print(len(good), len(normal), len(bad))
        return result

    def pop(self, num):
        num = int(num / 3)
        if num < len(self.container_good):
            for i in range(num):
                self.container_good.popleft()
        if num < len(self.container_bad):
            for i in range(num):
                self.container_bad.popleft()

        if num < len(self.container_normal):
            for i in range(num):
                self.container_normal.popleft()

    def size(self):
        return len(self.container_good) + len(self.container_normal) + len(self.container_bad)
