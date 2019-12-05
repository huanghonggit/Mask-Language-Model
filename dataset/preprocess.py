import nltk
import html
import re
import multiprocessing as mp
from multiprocessing import Pool
import tqdm
import sys

in_path = 'test.txt'
out_path = 'test_pre.txt'


_logits = '1234567890'
_punctuation = '\'(),.:;? $*=!/"\&-#'
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
symbols = list(_logits) + list(_letters) + list(_punctuation)
# print(len(symbols))


other_char = []


# def preprocess(in_path, out_path):
def preprocess(line):

    lines = []
    # 将每行以句子为单位分开
    sents = nltk.sent_tokenize(line)
    # 遍历每个句子
    for sent in sents:

        is_hanzi = has_hanzi(sent)  # 判断是否存在汉字
        is_pure_english = judge_pure_english(sent.strip())  # 判断是否是纯英文 （字母 数字 符号）

        if is_hanzi or not is_pure_english: # 是汉字或者不是纯英文时会执行
            # other_lines.append(sent)
            continue

        if 30 < len(sent) and len(sent) < 512: # 判断长度, 截取前512字符

            sent = html.unescape(sent.strip())  # 将字符串中的html实体转换成html标签

            # sent = ' '.join(nltk.word_tokenize(sent)).strip() + '\n'  # 先将句子进行tokenize,然后在token之间加空格，最后都转成小写
            lines.append(sent + '\n')

    return len(sents), lines


def has_hanzi(line):
  zhmodel = re.compile(u'[\u4e00-\u9fa5]')
  match = zhmodel.search(line)
  if match:
    return True
  else:
    return False


def judge_pure_english(keyword):
    return all(ord(c) < 128 for c in keyword)

def html_unescape(_str):  # 将字符串中的html实体转换成html标签
    return html.unescape(_str)

def progbar(i, n, size=55):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar

def stream(message):
    sys.stdout.write(f"\r{message}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    args = parser.parse_args()

    p = Pool(mp.cpu_count() - 1)
    print(mp.cpu_count() - 1)

    results = []

    data = open(args.input_path, 'r', encoding='utf-8').readlines()
    out_lines = []
    count = 0
    for line in data:
        # 一个进程执行一句
        res = p.apply_async(preprocess, (line,))
        results.append(res)

    p.close()
    p.join()

    for i, res in enumerate(results):

        line_count, lines = res.get()  # linecount表示一个线程执行一句， 这一句分出的数量； lines表示这一句最后的输出
        out_lines += lines
        count += line_count

        bar = progbar(i, len(data))
        message = f'{bar} {i}/{len(data)} '
        stream(message)

    sort_lines = sorted(out_lines, key=lambda i: len(i), reverse=True)

    print("原始总共：{}句，分句后{}，删除{}句，剩余{}句！".format(len(data), count, count-len(sort_lines), len(sort_lines)))
    # 预处理后的句子写入到新的文件中去
    with open(args.output_path, 'w') as f:
        f.writelines(sort_lines)

