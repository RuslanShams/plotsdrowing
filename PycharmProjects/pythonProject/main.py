import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_json('https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json')
# df = df.rename(columns={'mean': 'floor_vs_cailing_mean', 'max': 'floor_vs_cailing_max', 'min': 'floor_vs_cailing_min'})

# total_mean = df['floor_vs_cailing_mean'].mean()
# total_mean = round(total_mean, 6)
# total_max = df['floor_vs_cailing_max'].mean()
# total_max = round(total_max, 6)
# total_min = df['floor_vs_cailing_min'].mean()
# total_min = round(total_min, 6)

# floor_mean_dispersion = (max(df['floor_mean']), min(df['floor_mean']))
# floor_mean_dispersion = tuple(map(lambda x: round(x, 6), floor_mean_dispersion))

# floor_max_dispersion = (max(df['floor_max']), min(df['floor_max']))
# floor_max_dispersion = tuple(map(lambda x: round(x, 6), floor_max_dispersion))
#
# floor_min_dispersion = (max(df['floor_min']), min(df['floor_min']))
# floor_min_dispersion = tuple(map(lambda x: round(x, 6), floor_min_dispersion))
#
#
# ceiling_mean_dispersion = (max(df['ceiling_mean']), min(df['ceiling_mean']))
# ceiling_mean_dispersion = tuple(map(lambda x: round(x, 6), ceiling_mean_dispersion))
#
# ceiling_max_dispersion = (max(df['ceiling_max']), min(df['ceiling_max']))
# ceiling_max_dispersion = tuple(map(lambda x: round(x, 6), ceiling_max_dispersion))
#
# ceiling_min_dispersion = (max(df['ceiling_min']), min(df['ceiling_min']))
# ceiling_min_dispersion = tuple(map(lambda x: round(x, 6), ceiling_min_dispersion))



plt.title("Floor mean Dataset")
plt.ylim(floor_mean_dispersion[1] - 10, floor_mean_dispersion[0] + 10)
plt.scatter(x=df.index, y=df['floor_mean'])
plt.hlines(y=total_mean, xmin=0, xmax=len(df['floor_mean']), colors='r')
plt.show()

plt.title("Floor max Dataset")
plt.ylim(floor_max_dispersion[1] - 10, floor_max_dispersion[0] + 10)
plt.scatter(x=df.index, y=df['floor_max'])
plt.hlines(y=total_mean, xmin=0, xmax=len(df['floor_max']), colors='r')
plt.show()

plt.title("Floor min Dataset")
plt.ylim(floor_min_dispersion[1] - 10, floor_min_dispersion[0] + 10)
plt.scatter(x=df.index, y=df['floor_min'])
plt.hlines(y=total_mean, xmin=0, xmax=len(df['floor_min']), colors='r')
plt.show()



plt.title("ceiling mean Dataset")
plt.ylim(ceiling_mean_dispersion[1] - 10, ceiling_mean_dispersion[0] + 10)
plt.scatter(x=df.index, y=df['ceiling_mean'])
plt.hlines(y=total_mean, xmin=0, xmax=len(df['ceiling_mean']), colors='r')
plt.show()

plt.title("ceiling max Dataset")
plt.ylim(ceiling_max_dispersion[1] - 10, ceiling_max_dispersion[0] + 10)
plt.scatter(x=df.index, y=df['ceiling_max'])
plt.hlines(y=total_mean, xmin=0, xmax=len(df['ceiling_max']), colors='r')
plt.show()

plt.title("ceiling min Dataset")
plt.ylim(ceiling_min_dispersion[1] - 10, ceiling_min_dispersion[0] + 10)
plt.scatter(x=df.index, y=df['ceiling_min'])
plt.hlines(y=total_mean, xmin=0, xmax=len(df['ceiling_min']), colors='r')
plt.show()



# class TooManyMemoryForFourSlots(Exception):
#     pass
#
# class CPU:
#     def __init__(self, name, fr):
#         self.name = name
#         self.fr = fr
#
#
# class Memory:
#     def __init__(self, name, volume):
#         self.name = name
#         self.volume = volume
#
#
#
# class MotherBord:
#     def __init__(self, name, cpu, *memory):
#         if len(memory) > 4:
#             raise TooManyMemoryForFourSlots
#         self.name = name
#         self.cpu = cpu
#         self.total_mem_slots = 4
#         self.mem_slots = list(memory)
#
#
#     def get_config(self):
#         def make_str(mems):
#             string_lst = []
#             for mem in mems:
#                 string_lst.append(f'{mem.name}-{mem.volume}')
#             return ';'.join(string_lst)
#
#         return [f'Материнская плата:{self.name}',
#                 f'Центральный процессор:{self.cpu.name},{self.cpu.fr}',
#                 f'Слотов памяти:{self.total_mem_slots}',
#                 f'Память:{make_str(self.mem_slots)}']
#
#
# mb = MotherBord('MSI B450', CPU('ryzen 5', 3200), Memory('Kingston', 16), Memory('Micron', 32))

# class Graph:
#
#     def __init__(self, data, is_show=True):
#         self.data = data.copy()
#         self.is_show = is_show
#
#     def set_data(self, data):
#         self.data = data.copy()
#
#     def show_table(self):
#         if self.is_show:
#             print(*self.data)
#         else:
#             print('Отображение данных закрыто')
#
#     def show_graph(self):
#         if self.is_show:
#             print('Графическое отображение данных:', *self.data)
#         else:
#             print('Отображение данных закрыто')
#
#     def show_bar(self):
#         if self.is_show:
#             print('Столбчатая диаграмма:', *self.data)
#         else:
#             print('Отображение данных закрыто')
#
#     def set_show(self, fl_show):
#         self.is_show = fl_show
#
# data_graph = list(map(int, input().split()))
# gr = Graph(data_graph)
# gr.show_bar()
# gr.set_show(False)
# gr.show_table()
# class TriangleChecker:
#     def __init__(self, a, b, c):
#         self.a = a
#         self.b = b
#         self.c = c
#     def is_triangle(self):
#         if not all([True if type(i) in (int, float) and i > 0 else False for i in self.__dict__.values()]):
#             return 1
#         elif self.a + self.b <= self.c and self.b + self.c <= self.a and self.a + self.c <= self.b:
#             return 2
#         else:
#             return 3
#
# a, b, c = map(int, input().split())
# a = str(a)
# tr = TriangleChecker(a, b, c)
# print(tr.is_triangle())

# class Line:
#     def __init__(self, a, b, c, d):
#         self.sp = (a, b)
#         self.ep = (c, d)
#
# class Rect:
#     def __init__(self, a, b, c, d):
#         self.sp = (a, b)
#         self.ep = (c, d)
#
# class Ellipse:
#     def __init__(self, a, b, c, d):
#         self.sp = (a, b)
#         self.ep = (c, d)
#
# def zero_line(x):
#     if isinstance(x,  Line):
#         x.a = 0
#         x.b = 0
#         x.c = 0
#         x.d = 0
#         return x
#     else:
#         return x
# cl_set = {Line, Rect, Ellipse}
# digits = set(range(0, 500))
# elements = [list(cl_set).pop()(list(digits).pop(), list(digits).pop(), list(digits).pop(), list(digits).pop())
#             for _ in range(217)]
# elements = list(map(zero_line, elements))
# print(elements)


# class Point:
#     def __init__(self, x, y, color='black'):
#         self.x = x
#         self.y = y
#         self.color = color
# points = [Point(i, i) for i in range(1, 2000, 2)]
# points[1].color = 'yellow'

# class Translator:
#     def add(self, eng, rus):
#         if 'dic' not in self.__dict__:
#             self.dic = {}
#         self.dic.setdefault(eng, [])
#         if rus not in self.dic[eng]:
#             self.dic[eng].append(rus)
#     def remove(self, eng):
#         del self.dic[eng]
#
#     def translate(self, eng):
#         return self.dic[eng]
# tr = Translator()
# for pair_of_words in ('tree - дерево', 'car - машина', 'car - автомобиль',
#                       'leaf - лист', 'river - река', 'go - идти',
#                       'go - ехать', 'go - ходить', 'milk - молоко'):
#     tr.add(*pair_of_words.split(' - '))
# tr.remove('car')
# print(*tr.translate('go'))

# import sys
#
# # программу не менять, только добавить два метода
# lst_in = list(map(str.strip, sys.stdin.readlines()))  # считывание списка строк из входного потока
#
#
# class DataBase:
#     lst_data = []
#     FIELDS = ('id', 'name', 'old', 'salary')
#
#     def select(self, a, b):
#         try:
#             return self.lst_data[a:b+1]
#         except:
#             return self.lst_data[a:]
#
#     def insert(self, data):
#         [self.lst_data.append(dict(zip(self.FIELDS, i.split()))) for i in data]
#
#
# db = DataBase()
# db.insert(lst_in)
# print(db.select(1, 6))
# print(dict(zip([1, 'Сергей', 35, 120000], ['id', 'name', 'old', 'salary'])))

# import sys
#
# class StreamData:
#     def create(self, fields, lst_values):
#         pr
#         if len(fields) == len(lst_values):
#             return False
#         print('sdfsd')
#         for k, v in zip(fields, lst_values):
#             setattr(self, k, v)
#         return True
#
# class StreamReader:
#     FIELDS = ('id', 'title', 'pages')
#
#     def readlines(self):
#         lst_in = list(map(str.strip, sys.stdin.readlines()))  # считывание списка строк из входного потока
#         sd = StreamData()
#         res = sd.create(self.FIELDS, lst_in)
#         return sd, res
#
# sd = StreamData()
# sd.create(['id', 'name', 'comment'], [4, 'Имя', "Какой-то текст"])
# print(sd.__dict__)
# sr = StreamReader()
# data, result = sr.readlines()
# class Graph:
#     LIMIT_Y = [0, 10]
#
#     def set_data(self, data: list[int]):
#         self.data = data
#
#     def drow(self):
#         print(*[i for i in self.data if self.LIMIT_Y[0] <= i <= self.LIMIT_Y[1]])
#
# graph_1 = Graph()
# graph_1.set_data([10, -5, 100, 20, 0, 80, 45, 2, 5, 7])
# graph_1.drow()

# class MediaPlayer:
#     def open(self, file):
#         self.filename = file
#     def play(self):
#         print(f'Воспроизведение {self.filename}')
#
# media1 = MediaPlayer()
# media2 = MediaPlayer()
# media1.open('filemedia1')
# media2.open('filemedia2')
# media1.play()
# media2.play()

# class Person:
#     name = 'Сергей Балакирев'
#     job = 'Программист'
#     city = 'Москва'
#
# p1 = Person()
# print('job' in p1.__dict__.keys())

# class Figure:
#     type_fig = 'ellipse'
#     color = 'red'
# fig1 = Figure()
# dic = {'start_pt': (10, 5),
#     'end_pt': (100, 20),
#     'color': 'blue'}
# for k, v in dic.items():
#     setattr(fig1, k, v)
# del fig1.color
# print(*fig1.__dict__.keys())

# import re
# pstart, pend = list(map(int, input().split()))
# string = input()
# pattern = re.compile(r'\d+')
# finded = pattern.findall(string, pos=pstart, endpos=pend)
# int_list = list(map(int, finded))
# print(sum(int_list))
# def multiple_split(string, delimiters):
#     delim = '|'.join(delimiters)
#     pattern = ''
#     for ch in delim:
#         if ch != '|':
#             pattern += f'\\' + ch
#         else:
#             pattern += ch
#     return re.split(pattern, string)
# print(multiple_split('beegeek-python.stepik', ['.', '-']))
# print(multiple_split('Timur---Arthur+++Dima****Anri', ['---', '+++', '****']))
# print(multiple_split('timur.^[+arthur.^[+dima.^[+anri.^[+roma.^[+ruslan', ['.^[+']))
# s = input()
# s = re.split(r'\s*(?:and|or|[|&])\s*', s)
# print(*s, sep=', ')
# import sys
# st = sys.stdin.read()
# st = re.sub(r'(^| *) *\"\"\".*?\"\"\"\n', r'', st, flags=re.DOTALL)
# st = re.sub(r'[ ]{4,}#.+\n', r'', st)
# st = re.sub(r'( {2}#.+\n)', r'\n', st)
# st = re.sub(r'\n#.+\n', r'\n', st)
# st = re.sub(r'((^)|(\n))#.+', r'', st)
# print(st)
# s, n = input(), 1
#
# while n:
#     s, n = re.subn(r'(\b\w+)(\W+)\1\b', r'\1', s)
#     print(s)
#
# print(s)


# def multi(str):
#     str = str[1]
#     digit = int(re.search(r'(^\d*)', str)[1])
#     str = re.sub(r'(^\d*)', r'', str)
#     str = re.sub(r'^\(|\)$', r'', str)
#     return str * digit
# string = input()
# string = re.subn(r'(\d+\([^()]+\))', multi, string)
# while string[1] > 0:
#     string = re.subn(r'(\d+\([^()]+\))', multi, string[0])
# print(string[0])

# string = input()
# string = re.sub(r'\b(\w)(\w)(\w*)\b', r'\2\1\3', string)
# print(string)



# import keyword
#
# wrong_words = keyword.kwlist
# string = input()
# for i in wrong_words:
#     string = re.sub(fr'\b{i}\b', r'<kw>', string, flags=re.I)
# print(string)

# def normalize_whitespace(str):
#     return re.sub(r' +', ' ', str)
#
# print(normalize_whitespace('Тут   н   е   т     л   и     шних пробелов     '))


# def normalize_jpeg(str):
#     return re.sub(r'\.\w+$', r'.jpg', str, flags=re.I)
#
#
# print(normalize_jpeg('file.jepg.JPEG'))



# from mpmath import *
# w=lambda z1: 2*sqrt (9.8*(z1-0.09))/0.1
# plot (w,[0.1,0.16])

# jupyter notebook --no-browser --ip=0.0.0.0 --port=8080


#
# import pandas as pd
# from math import sqrt
# import matplotlib.pyplot as plt
#
# df = pd.read_json('https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json')
# print(df['ceiling_mean'].std())
# df = df.rename(columns={'mean': 'forecast_mean', 'max': 'forecast_max', 'min': 'forecast_min'})
# lst = []
# for i in range(len(df)):
#     lst.append((df.gt_corners[i] - df.rb_corners[i])**2)
# print(sqrt(pd.DataFrame(lst).mean()))



# import matplotlib.pyplot as plt
# df = pd.read_json('https://ai-process-sandy.s3.eu-west-1.amazonaws.com/purge/deviation.json')
# r = pd.DataFrame(df.iloc[0])


# game_data = {
#     'game_id': range(1, 6),        # id игрульки
#     'title': ('LOL', 'WH40K',
#               'WOW', 'POE',
#               'KEK'),              # наименование игрульки
#     'author_id': (1, 1, 2, 3, 5),  # id автора
#     'genre_id': (1, 2, 4, 4, 1),   # id жанра
#     'price': (0, 0, 14, 19, 100)   # цена игрульки
# }
#
# author_data = {
#     'author_id': (1, 2, 3, 4),     # id автора
#     'author_name': ('Graham McNeill',
#                     'Christie Golden',
#                     'Nick Jones',
#                     'Slaik')       # имя автора
# }
# game = pd.DataFrame(game_data)
# author = pd.DataFrame(author_data)
# g_df = game.merge(author, how='outer', on='author_id')
# print(g_df.loc[((g_df.price > 0) & (g_df.author_name.notnull())), ['author_name', 'price']].astype({'price': np.int8}))

# genre_data = {
#     'genre_id': (1, 2, 3),                 # id жанра
#     'genre_name': ('MOBA','RTS','RPG')     # название жанра
# }
#
# game_data = {
#     'game_id': range(1, 6),                # id игрульки
#     'title': ('LOL', 'WH40K',
#               'WOW', 'POE',
#               'KEK'),                      # наименование игрульки
#     'author_id': (1, 1, 2, 3, 5),          # id автора
#     'genre_id': (1, 2, 4, 4, 1),           # id жанра
#     'price': (0, 0, 14, 19, 100)            # цена игрульки
# }
# genre = pd.DataFrame(genre_data)
# game = pd.DataFrame(game_data)
# g_df = genre.merge(game, how='outer', on='genre_id')
# print(g_df.iloc[3].genre_name)

# teas_data = {'id': range(1, 7),
#              'name': ['Monkey King',
#                       'Long Yang',
#                       'Dammann',
#                       'GreenField',
#                       'Yunnan',
#                       'GunTin'],
#              'type_id': [1, 1, 2, 2, 3, 3]}
#
# types_data = {'id': range(1, 4),
#               'name': ['white', 'green', 'black']}
#
# teas = pd.DataFrame(teas_data)
# types = pd.DataFrame(types_data)
# g_df = teas.merge(types, how='outer', left_on='type_id', right_on='id')
# del g_df['id_x']
# del g_df['type_id']
# del g_df['id_y']
# g_df = g_df.rename({'name_x': 'name_tea', 'name_y': 'name_type'}, axis=1)
# print(g_df)


# df = pd.read_csv('https://stepik.org/media/attachments/course/105785/franchises.csv', encoding='cp1251')
# for i in range(len(df)):
#     df.loc[(i, 't_cost')] = df.loc[(i, 'cost')] + df.loc[(i, 'cost')] * df.loc[(i, 'commission')] / 100
# df['f_code'] = df['f_code'].apply(func=lambda x: x.replace('00', ''))
# df = df.groupby(by=['f_code', 'c_code', 'year']).agg(func={'t_cost': sum})
# maxi = df.groupby(by=['f_code'], as_index=False).agg(func={'t_cost': max})
# # print((maxi.iloc[0]))
# for i in range(len(maxi)):
#     p = df.loc[(df.t_cost == float(maxi.iloc[i]))]
#     print(f'{p.iloc[0].name[0]}-{p.iloc[0].name[1]}-{p.iloc[0].name[2]}-{float(maxi.iloc[i])}', end=';')
# print(maxi)
# df = df.cost.sum()
# df = df.groupby(by='f_code').agg(func=[max])
#
# df.to_dict()
# for k, v in df.items():
#     v.to_dict()
#     for f, g in v.items():
#         print(f'{f}:{round(g)}', end=';')

#
# data = {
#     'col_1': ['a', 'b'] * 9,
#     'col_2': [98, 99] * 9,
#     'col_3': [x * 2 for x in range(18)],
#     'col_4': [y * 5 / 100 for y in range(18)]
# }
# df = pd.DataFrame(data)
# df['col_2'] = df['col_2'].apply(func=lambda x: x - 1)
# df = df.groupby(by=['col_1','col_2']).agg(func={'col_3': ['sum'], 'col_4': ['mean']})
#
# print(df)
# df = pd.read_csv('https://stepik.org/media/attachments/course/105785/franchises.csv', encoding='cp1251')
# df = dfd.groupby(by=['store_name'], as_index=False)
# dfsum = pd.DataFrame(df.income.sum())
#
# print(dfsum)

# data = {
#     'col_1': ['o', 'e', 'e', 'e', 'o', 'o', 'o'],
#     'col_2': [1, 2, 4, 6, 3, 5, 7],
#     'col_3': [1, 0, 0, 0, 1, 1, 1]
# }
# df = pd.DataFrame(data)
# df = df.sort_values(by=['col_1', 'col_2'], ascending=[True, False])
# print(df)

# data = {
#     'col_1': [1, 1, 0, 1, 0, 0, 1],
#     'col_2': [True, True, False, True, False, False, True]
# }
# df = pd.DataFrame(data)
# dfg = df.groupby(by='col_1').count()
# d = pd.DataFrame(dfg)
# d = d.rename(columns={'col_2': 'count'})
# print(d)



# df = pd.read_csv('https://stepik.org/media/attachments/course/105785/stores.csv', encoding='cp1251')
# # dfmaxscore = df.score.max()
# print(df.country.value_counts(ascending=True).iloc[0])
# # dfmed = df.median(numeric_only=True)
# # dfme = df.mean(numeric_only=True)
# # res = dfme - dfmed
# # res.to_dict()
# # for i, v in res.items():
# #     print(f'{i}: {round(v * 10**6, 2)}', end=';')

# data = dict()
# for _ in range(3):
#     k, *v = input().split()
#     try:
#         v = list(map(int, v))
#     except ValueError:
#         pass
#     data[k] = v
#
#
# df = pd.DataFrame(data)
#
# mean_score = df.score.mean()
# max_score = df.loc[df.score == df.score.max(), ['user_name', 'score']]
# min_score = df.loc[df.score == df.score.min(), ['user_name', 'score']]
#
# print(f'mean score is {mean_score}\nmax score:\n{max_score}\nmin score:\n{min_score}')

# import pandas as pd
# import numpy as np
#
# user_id = input().split()
# user_name = input().split()
# score = input().split()
# score = list(map(int, score[1:]))
# dic = {}
# dic[user_id[0]] = user_id[1:]
# dic[user_name[0]] = user_name[1:]
# dic['score'] = score
# df = pd.DataFrame(dic)
# del df['user_id']
# print(f'mean score is {df.mean(axis=0, skipna=True, numeric_only=True).score}')
# maximum = df.loc[df.score.isin(df.max(axis=0, skipna=True, numeric_only=True))]
# minimum = df.loc[df.score.isin(df.min(axis=0, skipna=True, numeric_only=True))]
# print('max score:')
# print(maximum)
# print('min score:')
# print(minimum)

# print(f'mean score is {df.score.mean()}')
# print('max score:')
# print(df.max())
# print(df.loc[int(df.max().index):int(df.max().index), ['user_name', 'score']])
# print('min score:')
# print(df.loc[int(df.min().user_id)-1:int(df.min().user_id)-1, ['user_name', 'score']])


# data = {
#     'col_1': range(1,10),
#     'col_2': ('K1', 'LO23', '11PR', 'RU2', 'K2', 'K3', 'P2', 'LOL11', '12PO'),
#     'col_3': (7, 8, 11, 2, None, 4, 2, 9, 7),
#     'col_4': (1000.2, 1182.2, 1234.5, 6543.6, None, 6665.4, 2344.5, 3873.9, 3323.4)
# }
# df = pd.DataFrame(data).describe(include='all')
# print(df.loc['std'])


# df = pd.read_csv('https://stepik.org/media/attachments/course/105785/user_apps.csv')
# temp_df = df.loc[(pd.notna(df.app_name))]
# # temp_df = temp_df.loc[(temp_df.email.str.endswith('.com'))]
# temp_df = temp_df.loc[(temp_df.app_version.str[0] >= '3')]
# print(temp_df.iloc[(99, 3)])
#
# data = {'col_1': range(1, 100),
#         'col_2': range(100, 199),
#         'col_3': ['yummy']*99,
#         'col_4': ['python']*99
#        }
#
# df = pd.DataFrame(data)
# df['col_5'] = df['col_3'] + ' ' + df['col_4']
# print(df.col_5.head())
# data = [input().split(';') for _ in range(5)]
# df = pd.DataFrame(data[1:], columns=data[0], index=range(1, len(data)))
# df = df.astype({'id': np.int8, 'income': np.int32, 'tax': np.int8, 'year': np.int16})
#
# for i in range(1,5):
#     df.loc[(i, 'income')] = df.loc[(i, 'income')] - df.loc[(i, 'income')] * df.loc[(i, 'tax')] / 100
#
# del df['tax']
# print(df)


# def change_me(data, changes):
#     df = pd.DataFrame(data)
#     for i, col, val in changes:
#         df.loc[(i, col)] = val
#     return df
# data = {'name': ['Python', 'Java', 'PHP'], 'version': [2, 'SE8', 8], 'year': [1999, 2010, 2015]}
# changes = [(0, 'year', 2000), (1, 'year', 2014)]

# print(change_me(data, changes))
# album = input().split()
# release = [np.int16(i) for i in input().split()]
# nsongs = [np.int8(i) for i in input().split()]
# data = zip(album, release, nsongs)
# data_f = pd.DataFrame(data, columns=['album_name', 'release_year', 'nsongs'])
# data_f.index = data_f.index + 1
# print(data_f)

# data = {'columns': ['fullname', 'age', 'hobby'],
#         'data': [['Tatyana Ivanovna', 87, 'Мotocross'],
# ['Nina Vasilievna', 72, 'Stand up'],
# ['Olga Alekseevna', 69, 'TV'],
# ['Zoya Stepanovna', 79, 'TV'],
# ['Olga Petrovna', 73, 'Coocking'],
# ['Claudia Andreevna', 75, 'Politics']]
#  }
# def babushka(data):
#     return pd.DataFrame(data['data'], columns=data['columns'], index=list(range(1, len(data['data'])+1)))
# print(babushka(data))
# print(list(range(1, len(data['data']))))
#
# data = {'nlegs': [4, 4, 4],
#         'can_be_human': [False, False, False]}
#
# df = pd.DataFrame(data)
# df['nlegs'] = [2, 2, 2]
# df['can_be_human'] = True
# print(*df.can_be_human, end='')

# import matplotlib.pyplot as plt
#
# plt.close("all")
#
# data = pd.Series([int(i) for i in input().split()], name='scores')
# data_dic = pd.Series({i: j for i, j in zip(data.index, data) if j > 80}, name='scores')
# data_dic = data_dic.cumsum()
# data_dic.plot()
# def summa(d1, d2):
#     first = pd.Series(d1)
#     second = pd.Series(d2)
#     return first + second
#
# print(summa((1, 1, 1, 1), (2, 2, 2, 2)))

# dic = {i: int(j) for i, j in zip(input().split(';'), input().split(';'))}
# print(pd.Series(dic, name='age').astype(dtype=np.int8))


# def head_tail(data, part):
#     dt = pd.Series(data, name='names')
#     return dt[:3] if part == 'h' else dt[-3:]
# data = input().split()
# part = input()
# print(head_tail(data, part))

# import pandas as pd
#
# data = {'a': 97, 'b': 98, 'c': 99, 'd': 100, 'e': 101}
# s = pd.Series(data)
# result = s[['a', 'b', 'c']] + s[['c', 'd', 'e']]
# print(*result.array, end='')
import re
# from sys import stdin
# from collections import defaultdict
# from re import findall
#
# strings = [i.strip() for i in stdin]
# dic = defaultdict(list)
# for string in strings:
#     whats_find_adress = findall(r'<(\w*\b)', string)
#     for i in whats_find_adress:
#         dic[i]
#         line = findall(fr'{i} .*?>', string)
#         if line:
#             dic[i] = findall(r'\b(\S*)=', line[0])
#
# for k in sorted(dic.keys()):
#     v = sorted(dic[k])
#     print(f'{k}:', ', '.join(v))


# import re
# from re import findall
#
# def abbreviate(p):
#     return ''.join(findall(r'\b\w|[A-Z]', p)).upper()
#
#
# print(abbreviate('JS game sec'))


# import re
# from re import findall
#
# article = '''Stepik (до августа 2016 года Stepic) — это образовательная платформа и конструктор онлайн-курсов!
#
# Первые образовательные материалы были выпущены на Stepik 3 сентября 2013 года.
# В январе 2016 года Stepik выпустил мобильные приложения под iOS и Android. В 2017 году разработаны мобильные приложения для изучения ПДД в адаптивном режиме для iOS и Android...
#
# На октябрь 2020 года на платформе зарегистрировано 5 миллионов пользователей!
# Stepik позволяет любому зарегистрированному пользователю создавать интерактивные обучающие уроки и онлайн-курсы, используя видео, тексты и разнообразные задачи с автоматической проверкой и моментальной обратной связью.
#
# Проект сотрудничает как с образовательными учреждениями, так и c индивидуальными преподавателями и авторами.
# Stepik сегодня предлагает онлайн-курсы от образовательных организаций, а также индивидуальных авторов!
#
# Система автоматизированной проверки задач Stepik была использована в ряде курсов на платформе Coursera, включая курсы по биоинформатике от Калифорнийского университета в Сан-Диего и курс по анализу данных от НИУ «Высшая школа экономики»...
#
# Stepik также может функционировать как площадка для проведения конкурсов и олимпиад, среди проведённых мероприятий — отборочный этап Олимпиады НТИ (2016—2020) (всероссийской инженерной олимпиады школьников, в рамках программы Национальная технологическая инициатива), онлайн-этап акции Тотальный диктант в 2017 году, соревнования по информационной безопасности StepCTF-2015...'''
#
# print(len(findall(r'^Stepik', article, re.I | re.M)))
# print(len(findall(r'\.\.\.$|!$', article, re.I | re.M)))

# import re
# from sys import stdin
# from re import search, match, fullmatch
#
# strings = [i.strip() for i in stdin]
# score = 0
# for string in strings:
#     if search(r'.*beegeek.*', string, re.I | re.S):
#         score += 1
#
# print(score)



# import re
# from re import search, match, fullmatch
#
# string = input()
# if search(r'^Здравствуйте|^Доброе утро|^Добрый день|^Добрый вечер', string, re.I):
#     print(True)
# else:
#     print(False)
#


# from sys import stdin
# from re import search, match, fullmatch
#
# patterns = {3: r'^beegeek.*beegeek$', 2: r'^beegeek.*|.*beegeek$', 1: r'.+beegeek.+'}
# strings = [i.strip() for i in stdin]
# score = 0
# for string in strings:
#     for k, v in patterns.items():
#         if search(v, string):
#             score += k
#             print(k, string)
#             break
#
# print(score)


# from sys import stdin
# from re import search, match, fullmatch
#
# pattern = r'.*(bee).*(bee).*'
# pattern2 = r'.* (geek) .*|\b(geek) .*|.* (geek)\b|\b(geek)\b'
# strings = [i.strip() for i in stdin]
# c_bee = 0
# c_geek = 0
# for string in strings:
#     m = search(pattern, string)
#     n = search(pattern2, string)
#     if m and len(m.groups()) >= 2:
#         c_bee += 1
#     if n:
#         c_geek += 1
#
# print(c_bee)
# print(c_geek)
# from sys import stdin
# from re import search, match, fullmatch
#
# pattern = r'(\w+)(\1)'
# strings = [i.strip() for i in stdin]
# for string in strings:
#     if fullmatch(pattern, string):
#         print(string)

# from sys import stdin
# from re import search, match, fullmatch
#
# pattern = r'_\d+\w*_?'
# strings = [i.strip() for i in stdin]
# for string in strings:
#     print(True if fullmatch(pattern, string) else False)


# numbers = [i.strip() for i in stdin]
# for number in numbers:
#     match1 = fullmatch(r'(\d{1,3})([- ])(\d{1,3})(\2)(\d+)', number)
#     print(f'Код страны: {match1.group(1)}, Код города: {match1.group(3)}, Номер: {match1.group(5)}')



# def phone_num(num):
#     if num[0] == '7':
#         if num[1] != '-':
#             return False
#         if not num[2].isdigit() or not num[3].isdigit() or not num[4].isdigit() or num[5] != '-' or not num[6].isdigit() or not num[7].isdigit() or not num[8].isdigit() or num[9] != '-' or not num[10].isdigit() or not num[11].isdigit() or num[12] != '-' or not num[13].isdigit() or not num[14].isdigit():
#             return False
#     else:
#         if num[1] != '-':
#             return False
#         if not num[2].isdigit() or not num[3].isdigit() or not num[4].isdigit() or num[5] != '-' or not num[6].isdigit() or not num[7].isdigit() or not num[8].isdigit() or not num[9].isdigit() or num[10] != '-' or not num[11].isdigit() or not num[12].isdigit() or not num[13].isdigit() or not num[14].isdigit():
#             return False
#     return True
#
# def find(s):
#     numbers = []
#     for i in range(len(s)):
#         if s[i] == '7' or s[i]== '8':
#             if phone_num(s[i: i+16]):
#                 numbers.append(s[i: i+15])
#     return numbers
# s = input()
# print(*find(s), sep='\n')


# from string import ascii_lowercase
# from itertools import product
#
# # d = {0: "0", 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F'}
# lst = list(range(10)) + ['A', 'B', 'C', 'D', 'E',]
# n = int(input())
# m = int(input())
#
# f = product(lst[0:n+1], repeat=m)
# for i in f:
#     print(*i, sep='', end=' ')
#


# def password_gen():
#     for time in product(range(), range(10)):
#         yiel
#
#
# password_gen()
# for letter in letters:
#     for digit in digits:
#         print(letter, digit, sep='', end=' ')



# from collections import namedtuple
# import itertools as it
#
# Item = namedtuple('Item', ['name', 'mass', 'price'])
#
# items = [Item('Обручальное кольцо', 7, 49_000),
#          Item('Мобильный телефон', 200, 110_000),
#          Item('Ноутбук', 2000, 150_000),
#          Item('Ручка Паркер', 20, 37_000),
#          Item('Статуэтка Оскар', 4000, 28_000),
#          Item('Наушники', 150, 11_000),
#          Item('Гитара', 1500, 32_000),
#          Item('Золотая монета', 8, 140_000),
#          Item('Фотоаппарат', 720, 79_000),
#          Item('Лимитированные кроссовки', 300, 80_000)]
# weight = int(input())
# sum_weight = 0
# res = []
# for i in range(1, len(items)+1):
#     for j in set(it.combinations(items, i)):
#         f = sum(g.mass for g in j)
#         if f > weight:
#             continue
#         else:
#             res.append(sorted(j, key=lambda x: x.name))
# max = -1
# index = -1
# if res:
#     for inx, d in enumerate(res):
#         ss = sum(i.price for i in d)
#         if ss > max:
#             max = ss
#             index = inx
#     print(*(i.name for i in res[index]), sep='\n')
# else:
#     print('Рюкзак собрать не удастся')

# w = input()
# per = set(permutations(w))
# for i in sorted(per):
#     print(*i, sep='')

# def ranges(numbers):
#     left = numbers[0]
#     pairs = pairwise(numbers)
#     res = []
#     right = -1
#     for l, r in pairs:
#         if l + 1 != r:
#             right = l
#             res.append(tuple([left, right]))
#             left = r
#     else:
#         right = r
#         res.append((left, right))
#
#     return res


# numbers = [1, 3, 5, 7]
#
# print(ranges(numbers))
# numbers = [1, 2, 3, 4, 5, 6, 7]
#
# print(ranges(numbers))
#
# numbers = [1, 2, 3, 4, 7, 8, 10]
#
# print(ranges(numbers))

# def group_anagrams(words):
#     words_sorted = sorted(words, key=sorted)
#     words_group = groupby(words_sorted, key=sorted)
#     for k, v in words_group:
#         yield tuple(v)
#
#
#
# groups = group_anagrams(['крона', 'сеточка', 'тесачок', 'лучик', 'стоечка', 'норка', 'чулки'])
#
# print(*groups)


# tasks = [('Отдых', 'поспать днем', 3),
#         ('Ответы на вопросы', 'ответить на вопросы в дискорде', 1),
#         ('ЕГЭ Математика', 'доделать курс по параметрам', 1),
#         ('Ответы на вопросы', 'ответить на вопросы в курсах', 2),
#         ('Отдых', 'погулять вечером', 4),
#         ('Курс по ооп', 'обсудить темы', 1),
#         ('Урок по groupby', 'добавить задачи на программирование', 3),
#         ('Урок по groupby', 'написать конспект', 1),
#         ('Отдых', 'погулять днем', 2),
#         ('Урок по groupby', 'добавить тестовые задачи', 2),
#         ('Уборка', 'убраться в ванной', 2),
#         ('Уборка', 'убраться в комнате', 1),
#         ('Уборка', 'убраться на кухне', 3),
#         ('Отдых', 'погулять утром', 1),
#         ('Курс по ооп', 'обсудить задачи', 2)]
# tasks.sort(key=lambda x:x[0])
# grup = groupby(tasks, key=lambda x: x[0])
#
# for k, v in grup:
#     print(k + ':')
#     for i in sorted(v, key=lambda x: x[2]):
#         print(f'    {i[2]}. {i[1]}')
#     print()





# words = input().split()
# words.sort(key=len)
# grup = groupby(words, key=len)
# for k, v in grup:
#     print(k, end=' -> ')
#     print(*(sorted(i for i in v)), sep=', ')








# from collections import namedtuple
# from itertools import groupby
#
# Student = namedtuple('Student', ['surname', 'name', 'grade'])
#
# students = [Student('Гагиев', 'Александр', 10), Student('Дедегкаев', 'Илья', 11), Student('Кодзаев', 'Георгий', 10),
#             Student('Набокова', 'Алиса', 11), Student('Кораев', 'Артур', 10), Student('Шилин', 'Александр', 11),
#             Student('Уртаева', 'Илина', 11), Student('Салбиев', 'Максим', 10), Student('Капустин', 'Илья', 11),
#             Student('Гудцев', 'Таймураз', 11), Student('Перчиков', 'Максим', 10), Student('Чен', 'Илья', 11),
#             Student('Елькина', 'Мария', 11),Student('Макоев', 'Руслан', 11), Student('Албегов', 'Хетаг', 11),
#             Student('Щербак', 'Илья', 10), Student('Идрисов', 'Баграт', 11), Student('Гапбаев', 'Герман', 10),
#             Student('Цивинская', 'Анна', 10), Student('Туткевич', 'Юрий', 11), Student('Мусиков', 'Андраник', 11),
#             Student('Гадзиев', 'Георгий', 11), Student('Белов', 'Юрий', 11), Student('Акоева', 'Диана', 11),
#             Student('Денисов', 'Илья', 11), Student('Букулова', 'Диана', 10), Student('Акоева', 'Лера', 11)]
#
# students.sort(key=lambda x: x.name)
# groups = groupby(students, key=lambda x: x.name)
# max_result = max(groups, key=lambda tpl: sum(1 for i in tpl[1]))
# print(max_result[0])
# Person = namedtuple('Person', ['name', 'age', 'height'])
#
# persons = [Person('Tim', 63, 193), Person('Eva', 47, 158),
#            Person('Mark', 71, 172), Person('Alex', 45, 193),
#            Person('Jeff', 63, 193), Person('Ryan', 41, 184),
#            Person('Ariana', 28, 158), Person('Liam', 69, 193)]
# persons.sort(key=lambda x: x[0])
# persons.sort(key=lambda x: x[2])
# groups = groupby(persons, key=lambda x: x[2])
# for k, v in groups:
#     print(k, end=': ')
#     print(*(i.name for i in v), sep=', ')

# import itertools as it
#
# def grouper(itr, n):
#     s = iter(itr)
#     f = (s for _ in range(n))
#     yield from it.zip_longest(*f)
#
#
# iterator = iter([1, 2, 3, 4, 5, 6, 7])
#
# print(*grouper(iterator, 3))


# def ncycles(itr, times):
#     to_tee = it.tee(itr, times)
#     yield from it.chain(*to_tee)
#
# iterator = iter([1])
#
# print(*ncycles(iterator, 10))


# def max_pair(itr):
#     to_pair = it.pairwise(itr)
#     to_pair_sum = map(lambda x: x[0] + x[1], to_pair)
#     return max(to_pair_sum)

# iterator = iter([0, 0, 0, 0, 0, 0, 0, 0, 0])
#
# print(max_pair(iterator))
# def is_rising(itr):
#     to_itr = iter(itr)
#     prev = next(to_itr)
#     for num in to_itr:
#         if prev >= num:
#             return False
#         prev = num
#     return True
#
# iterator = iter(list(range(100, 201)) + [200])
# print(is_rising(iterator))

# def sum_of_digits(itr):
#     to_str = map(str, itr)
#     to_chain = it.chain.from_iterable(to_str)
#     to_int = map(int, to_chain)
#     return sum(to_int)
#
# print(sum_of_digits([123456789]))



# iters = tee([1, 2, 3], 3)
# print(*iters)
#
# totals = map(lambda a, b, c: a + b + c, *iters)
#
# print(next(totals))
# print(next(totals))


# def first_largest(itr, num):
#     s = 0
#     f = False
#     for i in itr:
#         if i > num:
#             f = True
#             break
#         s += 1
#     return s if f else -1

#
# numbers = [10, 2, 14, 7, 7, 18, 20]
#
# print(first_largest(numbers, 11))

# def take_nth(itr, n):
#     try:
#         return next(it.islice(itr, n-1, n))
#     except:
#         return None
#
#
# numbers = [11, 22, 33, 44, 55]
#
# print(take_nth(numbers, 3))

# def take(itr, n):
#     yield from it.islice(itr, n)
#
#
#
#
# print(*take(range(10), 5))


# def first_true(itr, pred):
#     if pred:
#         try:
#             return next(it.filterfalse(lambda x: not pred(x), itr))
#         except:
#             return None
#     try:
#         return next(filter(None, itr))
#     except:
#         return None
#
#
#
# numbers = (0, 0, 0, 69, 1, 1, 1, 2, 4, 5, 6, 0, 10, 100, 200)
# numbers_iter = filter(None, numbers)
#
# print(first_true(numbers_iter, lambda num: num < 0))

# def drop_this(itr, obj):
#     yield from it.dropwhile(lambda x: x == obj, itr)
#
# numbers = [0, 0, 0, 1, 2, 3]
#
# print(*drop_this(numbers, 0))
# def drop_while_negative(itr):
#     yield from it.dropwhile(lambda x: x < 0, itr)
#
#
#
#
# numbers = [-3, -2, -1, 0, 1, 2, 3]
#
# print(*drop_while_negative(numbers))
# from itertools import islice
#
# print(*islice('beegeek', 2, 6))

# def roundrobin(*args):
#     if not args:
#         return []
#     amount = len(args)
#     iters = map(iter, args)
#     c = it.count()
#     for el in it.cycle(iters):
#         try:
#             yield next(el)
#             c = it.count()
#         except:
#             if next(c) == amount:
#                 break
#
#
# print(*roundrobin('abc', 'd', 'ef'))
# def alnum_sequence():
#     bukvi = (chr(i) for i in range(65, 91))
#     d = (i for i in it.cycle(enumerate(bukvi, 1)))
#         yield d
# alnum = alnum_sequence()
#
# print(*(next(alnum) for _ in range(55)))
# def factorials(n):
#     seq = iter(range(1, n+1))
#     yield from it.accumulate(range(1, n+1), lambda x, y: x * y)
#
# numbers = factorials(6)
#
# print(*numbers)
# def tabulate(func):
#     yield from (func(i) for i in it.count(1))
#
# func = lambda x: x
# values = tabulate(func)
#
# print(next(values))
# print(next(values))

# import itertools as it
# import time
#
# symbols = ['.', '-', "'", '"', "'", '-', '.', '_']
#
# for c in it.cycle(symbols):
#     print(c, end='')
#     time.sleep(0.05)

# from itertools import cycle
#
# for i in enumerate(cycle(['a', 'b', 'c']), 0):
# for i in enumerate(cycle(['a', 'b', 'c']), 0):
#     print(i)


# def around(itr):
#     if not itr:
#         return
#     z = iter(itr)
#     prev = None
#     current = next(z)
#     for i in z:
#         yield (prev, current, i)
#         prev, current = current, i
#     else:
#         yield (prev, current, None)
# numbers = [1, 2, 3, 4, 5]
#
# print(*around(numbers))

# def pairwise(itr):
#     if not itr:
#         return
#     z = iter(itr)
#     d = next(z)
#     for i in z:
#         yield (d, i)
#         d = i
#     else:
#         yield (d, None)
#
#
# numbers = []
#
# print(*pairwise(numbers))

# def with_previous(itr):
#     d = {}
#     x = None
#     for i in itr:
#         d[i] = x
#         x = i
#         yield (i, d.get(x))
#
# iterator = iter('stepik')
#
# print(*with_previous(iterator))
# def stop_on(itr, obj):
#     f = []
#     for i in itr:
#         if i == obj:
#             break
#         f.append(i)
#     yield from f
#
#
# iterator = iter('beegeek')
#
# print(*stop_on(iterator, 'a'))
# def unique(itr):
#     f = []
#     for i in itr:
#         if i not in f:
#             f.append(i)
#     yield from f
#
# iterator = iter('111222333')
# uniques = unique(iterator)
#
# print(next(uniques))
# print(next(uniques))
# print(next(uniques))

# def txt_to_dict():
#     with open('planets.txt', 'r', encoding='utf-8') as file:
#         file_lines = (line.rstrip() for line in file)
#         dict = (s.split(' = ')for s in file_lines)
#         while True:
#             try:
#                 b = next(dict)
#             except:
#                 break
#             d = {}
#             while len(b) != 1:
#                 d[b[0]] = b[1]
#                 try:
#                     b = next(dict)
#                 except:
#                     break
#             yield d
#
#
#
# planets = txt_to_dict()
#
# for i in planets:
#     print(i)


# def nonempty_lines(file):
#     with open(file, 'r', encoding='utf-8') as filel:
#         file_lines = (line.rstrip() for line in filel if not line.isspace())
#         yield from (line if len(line) <= 25 else '...' for line in file_lines)
#
#
# lines = nonempty_lines('file1.txt')
#
# print(next(lines))
# print(next(lines))
# print(next(lines))
# print(next(lines))
# print(next(lines))

# from datetime import date, timedelta
#
# def years_days(year):
#     current_date = date(year=year, month=1, day=1)
#     while current_date.year == year:
#         yield current_date
#         current_date += timedelta(days=1)
#
# dates = years_days(2022)
#
# print(next(dates))
# print(next(dates))
# print(next(dates))
# print(next(dates))

# with open('data.csv', 'r', encoding='utf-8') as file:
#     file_lines = (line for line in file)
#     line_values = (line.rstrip().split(',') for line in file_lines)
#     file_headers = next(line_values)
#     line_dicts = (dict(zip(file_headers, data)) for data in line_values)
#
#     result = (
#         int(line['raisedAmt'])
#         for line in line_dicts
#         if 'a' == line['round']
#         )
#     print(sum(result))


# def filter_names(names, ignore_char, max_names):
#     filtred = filter(lambda x: x[0].lower() != ignore_char.lower(), names)
#     no_digits = (i for i in filtred if not any(map(str.isdigit, i)))
#     couner = 0
#     while couner < max_names:
#         try:
#             yield  next(no_digits)
#         except:
#             break
#         couner += 1
#
# data = ['Di6ma', 'Ti4mur', 'Ar5thur', 'Anri7620', 'Ar3453ina', '345German', 'Ruslan543', 'Soslanfsdf123', 'Geo000000r']
#
# print(*filter_names(data, 'A', 100))
# def parse_ranges(ranges):
#     diap = (i.split('-') for i in ranges.split(','))
#     ran = (range(int(i[0]), int(i[1]) + 1) for i in diap)
#     yield from iter(g for i in ran for g in i)
#
# print(*parse_ranges('1-2,4-4,8-10'))

# from collections import namedtuple
#
# Person = namedtuple('Person', ['name', 'nationality', 'sex', 'birth', 'death'])
#
# persons = [Person('E. M. Ashe', 'American', 'male', 1867, 1941),
#            Person('Goran Aslin', 'Swedish', 'male', 1980, 0),
#            Person('Erik Gunnar Asplund', 'Swedish', 'male', 1885, 1940),
#            Person('Genevieve Asse', 'French', 'female', 1949, 0),
#            Person('Irene Adler', 'Swedish', 'female', 2005, 0),
#            Person('Sergio Asti', 'Italian', 'male', 1926, 0),
#            Person('Olof Backman', 'Swedish', 'male', 1999, 0),
#            Person('Alyson Hannigan', 'Swedish', 'female', 1940, 1987),
#            Person('Dana Atchley', 'American', 'female', 1941, 2000),
#            Person('Monika Andersson', 'Swedish', 'female', 1957, 0),
#            Person('Shura_Stone', 'Russian', 'male', 2000, 0),
#            Person('Jon Bale', 'Swedish', 'male', 1950, 0)]
# non_deth = (i for i in persons if i.death != 0 and i.nationality == 'Swedish' and i.sex == 'male')
# yanger = max(non_deth, key=lambda x: x.birth)
# print(yanger.name)
# def interleave(*args):
#     return (
#         elemts[i]
#         for i in range(len(args[0]))
#         for elemts in args
#     )
#
# numbers = [1, 2, 3]
# squares = [1, 4, 9]
# qubes = [1, 8, 27]
#
# print(*interleave(numbers, squares, qubes))


# def all_together(*args):
#     return (i for j in args for i in j)
#
# objects = [range(3), 'bee', [1, 3, 5], (2, 4, 6)]
#
# print(*all_together(*objects))

# def count_iterable(iterable):
#     return sum(1 for _ in iterable)
#
#
# print(count_iterable([1, 2, 3, 4, 5]))

# def is_prime(current_num):
#     if current_num == 1:
#         return False
#     return any(current_num % div == 0 for div in range(2, current_num))
#
# print(is_prime(1))


# def flatten(nested_list):
#     if '__iter__' not in dir(nested_list):
#         yield nested_list
#     else:
#         for i in nested_list:
#             yield from flatten(i)
#
#
# generator = flatten([[1, 2], [[3]], [[4], 5]])
#
# print(*generator)
# def palindromes():
#     c = 1
#     while True:
#         if str(c) == str(c)[::-1]:
#             yield c
#         c += 1

# generator = palindromes()
# numbers = [next(generator) for _ in range(30)]
#
# print(*numbers)


# def palindromes(start=1):
#     if str(start) == str(start)[::-1]:
#         yield start
#     yield from palindromes(start + 1)
#
# generator = palindromes()
# numbers = [next(generator) for _ in range(30)]
#
# print(*numbers)

# print(list(str(5555)) == list(reversed(str(5555))))


# def card_deck(suit):
#     card_values = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "валет", "дама", "король", "туз"]
#     card_mast = ["пик", "треф", "бубен", "червей"]
#     card_mast.remove(suit)
#     while True:
#         yield from [v + ' ' + m for m in card_mast for v in card_values]
#
# generator = card_deck('треф')
# cards = [next(generator) for _ in range(40)]
#
# print(*cards)


# from datetime import date, timedelta
# def dates(start, count=None):
#     if count:
#         yield from [start + timedelta(days=x) for x in range(0, count)]
#     else:
#         while True:
#             yield start
#             try:
#                 start += timedelta(days=1)
#             except OverflowError:
#                 break

# generator = dates(date(9999, 1, 7))
#
# for _ in range(348):
#     next(generator)
#
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))
#
# try:
#     print(next(generator))
# except StopIteration:
#     print('Error')

# def reverse(seq):
#     for i in seq[::-1]:
#         yield i
# generator = reverse('beegeek')
#
# print(type(generator))
# print(*generator)
# def primes(left, right):
#     current_num = left
#     while current_num <= right:
#         f = True
#         for div in range(2, current_num):
#             if current_num % div == 0:
#                 f = False
#                 break
#         if f:
#             yield current_num
#         current_num += 1

# generator = primes(37, 37)
#
# try:
#     print(next(generator))
#     print(next(generator))
# except StopIteration:
#     print('Error')

# generator = primes(6, 36)
#
# print(next(generator))
# print(next(generator))

# def alternating_sequence(count=None):
#     current_num = 0
#     if not count:
#         while True:
#             current_num += 1
#             yield current_num if current_num % 2 != 0 else current_num * -1
#     else:
#         while current_num != count:
#             current_num += 1
#             yield current_num if current_num % 2 != 0 else current_num * -1
# generator = alternating_sequence(10)
#
# print(*generator)

# def simple_sequence():
#     counter = 0
#     while True:
#         counter += 1
#         for _ in range(counter):
#             yield counter
#
# generator = simple_sequence()
#
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))
# print(next(generator))
# class Xrange:
#     def __init__(self, start, stop, step=1):
#         self.start = start
#         self.stop = stop
#         self.step = step
#         self.res = None
#         self.b = start < stop
#     def __iter__(self):
#         return self
#     def __next__(self):
#         self.res = self.start
#         if self.b:
#             if self.res >= self.stop:
#                 raise StopIteration
#         else:
#             if self.res <= self.stop:
#                 raise StopIteration
#         self.start += self.step
#         if isinstance(self.step, float):
#             return float(self.res)
#         else:
#             return self.res




# xrange = Xrange(10, 1, -1)
#
# print(*xrange)

# xrange = Xrange(0, 3, 0.5)
#
# print(*xrange, sep='; ')
# class Alphabet:
#     def __init__(self, language):
#         self.language = language
#         self.ru = 'абвгдежзийклмнопрстуфхцчшщъыьэюя'
#         self.ru_iter = iter('абвгдежзийклмнопрстуфхцчшщъыьэюя')
#         self.en = 'abcdefghijklmnopqrstuvwxyz'
#         self.en_iter = iter('abcdefghijklmnopqrstuvwxyz')
#     def __iter__(self):
#         return self
#     def __next__(self):
#         if self.language == 'ru':
#             try:
#                 return next(self.ru_iter)
#             except:
#                 self.ru_iter = iter(self.ru)
#                 return next(self.ru_iter)
#         else:
#             try:
#                 return next(self.en_iter)
#             except:
#                 self.en_iter = iter(self.en)
#                 return next(self.en_iter)
#
#
# en_alpha = Alphabet('en')
#
# letters = [next(en_alpha) for _ in range(28)]
#
# print(*letters)


# from random import randint
# class RandomNumbers:
#     def __init__(self, left, right, n):
#         self.left = left
#         self.right = right
#         self.n = n
#         self.counter = -1
#     def __iter__(self):
#         return self
#     def __next__(self):
#         self.counter += 1
#         if self.counter == self.n:
#             raise StopIteration
#         return randint(self.left, self.right)
#
# iterator = RandomNumbers(1, 10, 2)
#
# print(next(iterator) in range(1, 11))
# print(next(iterator) in range(1, 11))


# class Cycle:
#     def __init__(self, iterable):
#         self.obj = iterable
#         self.itr = None
#         self.res = None
#     def __iter__(self):
#         return self
#     def __next__(self):
#         try:
#             self.res = next(self.itr)
#         except:
#             self.itr = iter(self.obj)
#             self.res = next(self.itr)
#         return self.res
#
# cycle = Cycle('be')
#
# print(next(cycle))
# print(next(cycle))
# print(next(cycle))
# print(next(cycle))

# class CardDeck:
#     def __init__(self):
#         self.mast = iter(("пик", "треф", "бубен", "червей"))
#         self.value = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "валет", "дама", "король", "туз")
#         self.dost = None
#         self.d = None
#         self.m = None
#     def __iter__(self):
#         return self
#     def __next__(self):
#         try:
#             self.d = next(self.dost)
#         except:
#             self.dost = iter(self.value)
#             self.m = next(self.mast)
#             self.d = next(self.dost)
#         return self.d + ' ' + self.m
#
#
#
# cards = CardDeck()
# for _ in range(100):
#     print(next(cards))




# class DictItemsIterator:
#     def __init__(self, data):
#         self.data = data
#         self.inx = -1
#     def __iter__(self):
#         return self
#     def __next__(self):
#         self.inx += 1
#         keys = list(self.data)
#         if self.inx == len(keys):
#             raise StopIteration
#         return keys[self.inx], self.data[keys[self.inx]]
#
# data = {1: 1, 2: 4, 3: 9, 4: 16, 5: 25, 6: 36, 7: 49}
#
# pairs = DictItemsIterator(data)
#
# print(*pairs)

# class PowerOf:
#     def __init__(self, number):
#         self.number = number
#         self.step = -1
#     def __iter__(self):
#         return self
#     def __next__(self):
#         self.step += 1
#         return self.number ** self.step
#
# power_of_two = PowerOf(2)
#
# print(next(power_of_two))
# print(next(power_of_two))
# print(next(power_of_two))
# print(next(power_of_two))


# class Fibonacci:
#     def __init__(self):
#         self.fb = [1, 1]
#         self.inx = -1
#     def __iter__(self):
#         return self
#     def __next__(self):
#         self.fb.append(self.fb[-1] + self.fb[-2])
#         self.inx += 1
#         return self.fb[self.inx]
# fibonacci = Fibonacci()
# for _ in range(10):
#     print(next(fibonacci))

# class Square:
#     def __init__(self, n):
#         self.limit = n
#         self.sq = 1
#         self.index = 1
#     def __iter__(self):
#         return self
#     def __next__(self):
#         if self.index < self.limit:
#             self.sq = self.index ** 2
#             self.index += 1
#             return self.sq
#         raise StopIteration
#
# squares = Square(10)
#
# print(list(squares))
# class BoundedRepeater:
#     def __init__(self, obj, times):
#         self.obj = obj
#         self.times = times
#         self.index = 0
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.index < self.times:
#             self.index += 1
#             return self.obj
#         else:
#             raise StopIteration
#
# bee = BoundedRepeater('bee', 2)
#
# print(next(bee))
# print(next(bee))
# numbers = [1, 2, 3, 4, 5]
#
# for i in numbers:
#     del numbers[0]
#     print(i)

# from random import choice
# def random_numbers(left, right):
#     return iter(lambda : choice(list(range(left, right+1))), 'ffff')
#
# iterator = random_numbers(1, 10)
#
# print(next(iterator) in range(1, 11))
# print(next(iterator) in range(1, 11))
# print(next(iterator) in range(1, 11))
# def is_iterator(obj):
#     if '__iter__' not in dir(obj):
#         return False
#     d = iter(obj)
#     return d is obj
# beegeek = map(str.upper, 'beegeek')
#
# print(is_iterator(beegeek))
# def is_iterable(obj):
#     print(dir(obj))
#     if '__iter__' in dir(obj):
#         return True
#     return False
#
#
# print(is_iterable('18731'))

# numbers = [1, 2, 3, 4, 5]
#
# if 'pop' in dir(numbers):
#     numbers.pop()
#
# print(numbers)
# def starmap(func, itr):
#     return map(lambda x: func(*x), itr)
#
# pairs = [(1, 3), (2, 5), (6, 4)]
#
# print(*starmap(lambda a, b: a + b, pairs))
# def get_min_max(itr):
#     if type(itr) is type(iter(range(1))):
#         return next(itr), max(itr)
#     f = sorted(itr)
#     if not f:
#         return None
#     return f[0], f[-1]
#
# data = iter(range(100_000_000))
#
# print(get_min_max(data))
# def get_min_max(data):
#     if not data:
#         return None
#     f = list(enumerate(data))
#     return min(f, key=lambda x: x[1])[0], max(f, key=lambda x: x[1])[0]
#
#
# data = []
#
# print(get_min_max(data))
# def transpose(matrix):
#     n = zip(*matrix)
#     return list(map(list, n))
#
#
#
# matrix = [[1, 2, 3, 4, 5],
#           [6, 7, 8, 9, 10]]
#
# for row in transpose(matrix):
#     print(row)
# def filterfalse(predicate, iterable):
#     if predicate is None:
#         return filter(lambda x: bool(x) is False, iterable)
#     return filter(lambda x: predicate(x) is False, iterable)
#
# numbers = [1, 2, 3, 4, 5]
#
# print(*filterfalse(lambda x: x >= 3, numbers))



# positive = (1, 2, 3)
# negative = map(lambda x: -x, positive)
# for a, b in zip(positive, negative):
#     print(None + None)

# non_zero = filter(None, [-2, -1, 0, 1, 2])
# positive = map(abs, non_zero)
#
# print(list(non_zero))
# print(list(positive))
# numbers = [100, 70, 34, 45, 30, 83, 12, 83, -28, 49, -8, -2, 6, 62, 64, -22, -19, 61, 13, 5, 80, -17, 7, 3, 21, 73, 88, -11, 16, -22]
#
# s = iter(numbers)
# numbers.clear()
# c = None
# while True:
#     try:
#         c = next(s)
#     except StopIteration:
#         break
# print(c)

# numbers = [100, 70, 34, 45, 30, 83, 12, 83, -28, 49, -8, -2, 6, 62, 64, -22, -19, 61, 13, 5, 80, -17, 7, 3, 21, 73, 88, -11, 16, -22]
# numbers = iter(numbers)
# for _ in range(3):
#     next(numbers)
# print(next(numbers))


# numbers = (-2, -1, 0, 1, 2)
#
# non_zero = filter(None, numbers)
# print(list(non_zero))


# import sys
#
# nums1 = [1, 2, 3]
# nums2 = nums1
# nums3 = [nums1, nums2]
#
# del nums1
#
# print(sys.getrefcount(nums2))

# from functools import lru_cache
# import sys
#
# def ways(n):
#     long = n
#     @lru_cache()
#     def rec(long):
#         if long == 1:
#             return 1
#         if long < 1:
#             return 0
#         return rec(long-1) + rec(long-3) + rec(long-4)
#     return rec(long)
#
# print(ways(100))
# @lru_cache()
# def sorting(word):
#     s = sorted(list(word))
#     return ''.join(s)
# [print(sorting(i.strip())) for i in open(0)]

# @lru_cache()
# def average(numbers):
#     return sum(numbers) / len(numbers)
#
# numbers = [1, 2, 3, 4, 5]
#
# print(average(numbers))
# print(average(numbers))


# from functools import partial
#
# def multiply(a, b):
#     '''Функция перемножает два числа и возвращает вычисленное значение.'''
#     return a * b
#
# double = partial(multiply, 2)
#
# print(double.__name__)
# print(double.__doc__)

# class MaxRetriesException(Exception):
#     pass
#
# def retry(times):
#     def decortor(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             counter = 0
#             while counter < times:
#                 try:
#                     return func(*args, **kwargs)
#                 except:
#                     counter += 1
#             else:
#                 raise MaxRetriesException
#         return wrapper
#     return decortor
#
#
# @retry(8)
# def beegeek():
#     beegeek.calls = beegeek.__dict__.get('calls', 0) + 1
#     if beegeek.calls < 5:
#         raise ValueError
#     print('beegeek')


# beegeek()

# def ignore_exception(*args):
#     ers = tuple(args)
#     def decortor(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             try:
#                 return func(*args, **kwargs)
#             except Exception as err:
#                 if type(err) in ers:
#                     print(f'Исключение {type(err).__name__} обработано')
#                 else:
#                     raise err
#
#         return wrapper
#     return decortor
#
#
# min = ignore_exception(ZeroDivisionError)(min)
#
# try:
#     print(min(1, '2', 3, [4, 5]))
# except Exception as e:
#     print(type(e))


# def add_attrs(**kwargs):
#     att = kwargs
#     def decortor(func):
#         for k, v in att.items():
#             func.__dict__[k] = v
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             return func(*args, **kwargs)
#         return wrapper
#     return decortor
#
#
# @add_attrs(attr2='geek')
# @add_attrs(attr1='bee')
# def beegeek():
#     return 'beegeek'
#
#
# print(beegeek.attr1)
# print(beegeek.attr2)
# print(beegeek.__name__)

# def takes(*args):
#     datatypes = list(args)
#     def decortor(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             for elem in args + tuple(kwargs.values()):
#                 if type(elem) not in datatypes:
#                     raise TypeError
#             return func(*args, **kwargs)
#         return wrapper
#     return decortor


# @takes(str)
# def beegeek(word, repeat):
#     return word * repeat
#
#
# try:
#     print(beegeek('beegeek', repeat=2))
# except TypeError as e:
#     print(type(e))

# @takes(int, str)
# def repeat_string(string, times):
#     return string * times
#
# print(repeat_string('bee', 3))

# @takes(list, bool, float, int)
# def repeat_string(string, times):
#     return string * times
#
# try:
#     print(repeat_string('bee', 4))
# except TypeError as e:
#     print(type(e))

# def returns(datatype):
#     def decortor(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             res = func(*args, **kwargs)
#             if not isinstance(res, datatype):
#                 raise TypeError
#             return res
#         return wrapper
#     return decortor
#
# @returns(list)
# def append_this(li, elem):
#     '''append_this docs'''
#     return li + [elem]
#
# print(append_this.__name__)
# print(append_this.__doc__)
# print(append_this([1, 2, 3], elem=4))


# def strip_range(start, end, ch='.'):
#     def decortor(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             l_str = list(func(*args, **kwargs))
#             for i in range(start, end):
#                 if i > len(l_str)-1:
#                     break
#                 l_str[i] = ch
#             return ''.join(l_str)
#         return wrapper
#     return decortor
#
#
# @strip_range(0, 1)
# def beegeek(word, end=" "):
#     """This is... Requiem. What you are seeing is indeed the truth"""
#     return word + end
#
# print(beegeek("beegee", end="k"))
# print(beegeek.__name__)
# print(beegeek.__doc__)

# def repeat(times):
#     def decortor(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             counter = times
#             def recur(counter):
#                 if counter
#                 return
#         return wrapper
#     return decortor
#
#
# @repeat(3)
# def say_beegeek():
#     '''documentation'''
#     print('beegeek')
#
#
# say_beegeek()



# def make_html(tag):
#     def decortor(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             return f'<{tag}>{func(*args, **kwargs)}</{tag}>'
#         return wrapper
#     return decortor
#
#
# @make_html('i')
# @make_html('del')
# def get_text(text):
#     return text
#
#
# print(get_text(text='decorators are so cool!'))

# import functools
#
# def prefix(ch, to_the_end=False):
#     def ch_r(func):
#         @functools.wraps(func)
#         def wrapper(*args, **kwargs):
#             if to_the_end:
#                 return func(*args, **kwargs) + ch
#             return ch + func(*args, **kwargs)
#         return wrapper
#     return ch_r
#
#
# @prefix('$$$', to_the_end=True)
# def get_bonus():
#     return '2000'
#
#
# print(get_bonus())


# def trace(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         print(f'TRACE: вызов {func.__name__}() с аргументами: {tuple(args)}, {dict(**kwargs)}')
#         res = func(*args, **kwargs)
#         print(f'TRACE: возвращаемое значение {func.__name__}(): {repr(res)}')
#         return res
#     return wrapper
#
#
# @trace
# def beegeek():
#     '''beegeek docs'''
#     return 'beegeek'
#
# print(beegeek())
# print(beegeek.__name__)
# print(beegeek.__doc__)


# def returns_string(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         res = isinstance(func(*args, **kwargs), str)
#         if not res:
#             raise TypeError
#         return func(*args, **kwargs)
#     return wrapper
#
#
# @returns_string
# def add(a, b):
#     return a + b
#
# try:
#     print(add(3, 7))
# except TypeError as e:
#     print(type(e))

# def square(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         return func(*args, **kwargs) ** 2
#     return wrapper
#
# @trace
# def add(a, b):
#     return a + b
#
# @square
# def add(a, b):
#     '''прекрасная функция'''
#     return a + b
#
# print(add(1, 1))
# print(add.__name__)
# print(add.__doc__)

# def make_capitalize(func):
#     @functools.wraps
#     def wrapper():
#         return func().capitalize()
#     return wrapper
#
# @make_capitalize
# def beegeek():
#     '''documentation'''
#     return 'beegeek'
#
# print(beegeek.__name__)
# print(beegeek.__doc__)

# def takes_positive(func):
#     def wrapper(*args, **kwargs):
#         res = list(reversed(args))
#         for v in kwargs.values():
#             res.append(v)
#         wrong_type = all([isinstance(i, int) for i in res])
#         wrong_value = all([True if i > 0 else False for i in res])
#         if not wrong_type:
#             raise TypeError
#         if not wrong_value:
#             raise ValueError
#         return func(*args, **kwargs)
#     return wrapper
#
#
# @takes_positive
# def positive_sum(*args, **kwargs):
#     return sum(args) + sum(kwargs.values())
#
#
# print(positive_sum(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, par1=1, sep=4))


# def exception_decorator(func):
#     def wrapper(*args, **kwargs):
#         try:
#             return func(*args, **kwargs), 'Функция выполнилась без ошибок'
#         except:
#             return None, 'При вызове функции произошла ошибка'
#     return wrapper
#
#
# sum = exception_decorator(sum)
#
# print(sum(['199', '1', 187]))



# def reverse_args(func):
#     def wrapper(*args, **kwargs):
#         res = list(reversed(args))
#         for v in kwargs.values():
#             res.append(v)
#         return func(*res)
#     return wrapper
#
#
# @reverse_args
# def power(a, n):
#     return a ** n
#
#
# print(power(2, 3))


# def do_twice(func):
#     def wrapper(*args, **kwargs):
#         res = func(*args, **kwargs)
#         func(*args, **kwargs)
#         return res
#     return wrapper
#
#
# @do_twice
# def beegeek():
#     print('beegeek')
#
#
# beegeek()

# @do_twice
# def beegeek():
#     return 'beegeek'
#
#
# print(beegeek())


# def p(func):
#     def wrapper(*args, **kwargs):
#         d = list(map(lambda x: str(x).upper(), args))
#         s = {k: v.upper() for k, v in kwargs.items()}
#         func(*d, sep=s.get('sep', ''), end=s.get('end', '\n'))
#     return wrapper
# dprint = p(print)
# dprint(111, 222, 333)

# def sandwich(func):
#     def wrapper(*args, **kwargs):
#         print('---- Верхний ломтик хлеба ----')
#         res = func(*args, **kwargs)
#         print('---- Нижний ломтик хлеба ----')
#         return res
#     return wrapper

# @sandwich
# def counter():
#     for i in range(17):
#         print(i)
#
# counter()

# @sandwich
# def add_ingredients(ingredients):
#     print(' | '.join(ingredients))
#
# add_ingredients(['томат', 'салат', 'сыр', 'бекон'])


# @sandwich
# def beegeek():
#     return 'beegeek'
#
#
# print(beegeek())

# def matrix_to_dict(matrix: list[list[int | float]]) -> dict[int, list[int | float]]:
#     return {i+1: matrix[i] for i in range(len(matrix))}
#
# annotations = matrix_to_dict.__annotations__
#
# print(annotations['return'])

# def cyclic_shift(numbers: list[int | float], step: int) -> None:
#     if step > 0:
#         for s in range(step):
#             buf = numbers[-1]
#             for i in reversed(range(len(numbers)-1)):
#                 numbers[i + 1] = numbers[i]
#             numbers[0] = buf
#     elif step < 0:
#         for s in range(abs(step)):
#             buf = numbers[0]
#             for i in range(1, len(numbers)):
#                 numbers[i - 1] = numbers[i]
#             numbers[-1] = buf
#
# numbers = [1, 2, 3, 4, 5]
# cyclic_shift(numbers, -2)
#
# print(numbers)


# def top_grade(grades: dict[str, str | list[int]]) -> dict[str, str | int]:
#     return {'name': grades['name'], 'top_grade': max(grades['grades'])}
#
#
# print(*top_grade.__annotations__.values())

# def get_digits(number: int | float) -> list[int]:
#     return [int(i) for i in str(number) if i != '.']
#
# annotations = get_digits.__annotations__
#
# print(annotations['return'])




# def sort_priority(values, group):
#     l_gr = list(group)
#     l_gr.sort()
#     l_v = values.copy()
#     l_v.sort()
#     for v in group:
#         if v not in values:
#             l_gr.remove(v)
#         else:
#             l_v.remove(v)
#     values.clear()
#     values.extend(l_gr)
#     values.extend(l_v)
#
#
# numbers = [8, 3, 1, 2, 5, 4, 7, 6]
# group = {5, 7, 2, 3}
# sort_priority(numbers, group)
#
# print(numbers)

# from datetime import date
# import locale
#
# def date_formatter(country_code):
#     s = {'ru': '%d.%m.%Y',
#          'us': '%m-%d-%Y',
#          'ca': '%Y-%m-%d',
#          'br': '%d/%m/%Y',
#          'fr': '%d.%m.%Y',
#          'pt': '%d-%m-%Y', }
#     def st(d):
#         return d.strftime(s[country_code])
#     return st
#
# date_ru = date_formatter('ca')
# today = date(2015, 12, 7)
# print(date_ru(today))

# def sourcetemplate(url):
#     def st(**kwargs):
#         u = url
#         f = False
#         if kwargs:
#             u += '?'
#         c = len(u)
#         for k in sorted(kwargs.keys()):
#             u += k + '=' + str(kwargs[k])
#             u += '&'
#             f = True
#         if f:
#             return u[:-1]
#         return u
#     return st
# url = 'https://all_for_comfort_life.com'
# load = sourcetemplate(url)
# print(load(smartphone='iPhone', notebook='huawei', sale=True))

# def generator_square_polynom(a, b, c):
#     def st(x):
#         return a * x ** 2 + b * x + c
#     return st
#
# f = generator_square_polynom(1, 2, 1)
# print(f(5))

# def power(degree):
#     def st(x):
#         return x ** degree
#     return st
#
#
# square = power(2)
# print(square(5))