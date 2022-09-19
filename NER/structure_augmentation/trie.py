#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/11/4 上午11:42
# @Author  : PeiP Liu
# @FileName: trie.py
# @Software: PyCharm
import collections


class TrieNode:
    def __init__(self):
        self.children = collections.defaultdict(TrieNode)
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root  # 每次都是从头开始插入
        for letter in word:
            current = current.children[letter]  # 如果曾经出现过该词, 就顺着向下走；否则新开辟字典分支
        current.is_word = True

    def search(self, word):
        current = self.root
        for letter in word:
            current = current.children.get(letter)  # 不断向下探索寻找
            if current is None:  # 无法向下查询，没有该词
                return False
        return current.is_word  # 正常查询到底，该词存在

    def startwith(self, prefix):
        current = self.root
        for letter in prefix:
            current = current.children.get(letter)
            if current is None:
                return False
        return True

    def enumerateMatch(self, word, space='_', backward=False):  # 该处用法未明
        matched = []
        while len(word) > 0:
            if self.search(word): # 可以在字典里查找到该词
                matched.append(space.join(word)) # word or word[:] ?
            del word[-1]
        return matched
