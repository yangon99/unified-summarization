# -*- coding: utf-8 -*-
from make_datafiles import *

'''
    这个文件用于处理 LCSTS 数据集。
'''

# 将中文词转换为整数
int2char = list()
char2int = dict()
char2int_index = 1

# LCSTS 数据集中的三部分
lcsts_train_steps = ['train', 'val', 'test']
lcsts_valid_counts = [2400591, 10666, 1106]
file_name_bias = 1

# 分词方式，word 表示使用 Stanford CoreNLP 做分词，需要使用其server模式
#         char 表示逐字读入
read_charc_type = "char"  # or word

def read_text_file_lcsts(data_file):
    '''
        从 LCSTS 文件中读取短文与摘要对，返回一个包含短文与摘要的元组
        (文章, 摘要)
    '''
    # all_stories = []
    with open(data_file, 'r') as fp:
        curr_arti = ""
        curr_abs = ""
        next_is_abs = False
        next_is_article = False
        for line in fp.readlines():
            if next_is_abs:
                curr_abs = line.strip()
                next_is_abs = False
            if next_is_article:
                curr_arti = line.strip()
                next_is_article = False
                yield (curr_arti, curr_abs)
                # all_stories.append((curr_arti, curr_abs))

            if "<summary>" in line:
                next_is_abs = True
            if "<short_text>" in line:
                next_is_article = True
    # return all_stories

def get_token(sent, parser):
    '''
        获取 sent 的分词，若 parser=="char" 则逐字，否则使用 CoreNLP 的server mode获取分词
    '''
    if parser != "char":
        return list(parser.tokenize(sent.decode('utf-8')))
    return [char for char in sent.decode('utf-8')]

def convert_char_to_int(words_list):
    '''
        输入分词后的单词/字(格式为 bytes), 对应返回其整数编码
        对应内容被保存在字典（dict） char2int 中，int2char 是一个列表（list），可用于反查
        such as: Input [u'你好', u'这', u'是', u'测试']
                 Return ['0', '1', '3', '4']
    '''
    global int2char
    global char2int
    global char2int_index
    ret = list()
    for cur_word in words_list:
        ret.append(str(char2int.get(cur_word.encode('utf-8'), char2int_index)))
        if ret[-1] == str(char2int_index):
            char2int[cur_word.encode('utf-8')] = char2int_index
            int2char.append(cur_word.encode('utf-8'))
            char2int_index += 1
    return ret

def tokenize_lcsts(stories_file, tokenized_stories_file, valid_count):
    '''
        使用 CoreNLP 分词或逐字分割 LCSTS 内容
        为了节省硬盘与内存开销，使用了生成器的方式进行处理
    '''
    global read_charc_type
    parser = "char"
    if read_charc_type == "word":
        parser = nltk.CoreNLPParser('http://127.0.0.1:9001')
    all_tokenized_stories = list()

    # print "Preparing to tokenize %s to %s" % (stories_file, tokenized_stories_file)
    all_origin_stories = read_text_file_lcsts(stories_file)
    index = 0
    for (cur_arti, cur_abs) in all_origin_stories:
        index += 1
        cur_ret = ' '.join(convert_char_to_int(get_token(cur_arti, parser)))
        cur_ret += '\n@highlight\n'
        cur_ret += ' '.join(convert_char_to_int(get_token(cur_abs, parser)))
        yield cur_ret
        # all_tokenized_stories.append(cur_ret)
        # if index % 10000 == 0:
            # print "currently resolved %d items" % index
    # if index != valid_count:
    #     raise Exception("The tokenized lcsts file %s contains %i files, but we got %i" %
    #                     (stories_file, valid_count, index))
    
    # with open(tokenized_stories_file, 'wb') as output_fp:
    #     pk.dump(all_tokenized_stories, output_fp)
    # print "Stanford CoreNLP Tokenizer has finished, files saved in %s" % (tokenized_stories_file)

def write_to_bin_lcsts(train_step, out_file):
    '''
        将数据处理后保存在 out_file 中
        当 train_step=="train" 的时候，会创建 vocab 文件
        大部分内容从原始文件中复制
    '''
    makevocab = False
    if train_step == 'train':
        makevocab = True

    print "Making bin file for %s files" % train_step
    print "The files will be saved in %s" % out_file
    # story_fnames = os.listdir(train_step)
    # num_stories = len(story_fnames)

    if makevocab:
        vocab_counter = collections.Counter()

    global extract_sents_num
    global extract_words_num
    global article_sents_num
    global extract_info
    extract_sents_num = []
    extract_words_num = []
    article_sents_num = []
    data = {'article': [], 'abstract': [], 'rougeLs': {'f': [], 'p': [],
                                                       'r': []}, 'gt_ids': [], 'select_ratio': [], 'rougeL_r': []}
    # with open(input_file, 'rb') as fp:
    #     all_stories = pk.load(fp)
    # num_stories = len(all_stories)
    idx = 0
    all_stories = tokenize_lcsts(os.path.join(lcsts_data_dir, data_file_name), os.path.join(lcsts_data_dir, data_file_name + '.tokenized.pkl'), lcsts_valid_counts[_count])
    with open(out_file, 'wb') as writer:
        for s in all_stories:
            idx += 1
            if idx % 1000 == 0:
                print "Writing story %i of %i; %.2f percent done" % (idx, lcsts_valid_counts[_count], float(idx) * 100.0 / float(lcsts_valid_counts[_count]))

            # story_file = os.path.join(train_step, s)
            # Get the strings to write to .bin file
            article_sents, abstract_sents, extract_sents, extract_ids, fs, ps, rs, max_Rouge_l_r = get_art_abs("#LCSTS" + s)
            ratio = float(len(extract_sents)) / len(article_sents) if len(article_sents) > 0 else 0

            # save scores of all article sentences
            data['article'].append(article_sents)
            data['abstract'].append(abstract_sents)
            data['rougeLs']['f'].append(fs)
            data['rougeLs']['p'].append(ps)
            data['rougeLs']['r'].append(rs)
            data['gt_ids'].append(extract_ids)
            data['select_ratio'].append(ratio)
            data['rougeL_r'].append(max_Rouge_l_r)

            # Make abstract into a signle string, putting <s> and </s> tags around the sentences
            article = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in article_sents])
            abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in abstract_sents])
            extract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in extract_sents])
            extract_ids = ','.join([str(i) for i in extract_ids])

            # Write to tf.Example
            tf_example = example_pb2.Example()
            tf_example.features.feature['article'].bytes_list.value.extend([article])
            tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
            tf_example.features.feature['extract'].bytes_list.value.extend([extract])
            tf_example.features.feature['extract_ids'].bytes_list.value.extend([extract_ids])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))

            # Write the vocab to file, if applicable
            if makevocab:
                art_tokens = article.split(' ')
                art_tokens = [t for t in art_tokens if t not in [
                    SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                abs_tokens = abstract.split(' ')
                abs_tokens = [t for t in abs_tokens if t not in [
                    SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                tokens = art_tokens + abs_tokens
                tokens = [t.strip() for t in tokens]  # strip
                tokens = [t for t in tokens if t != ""]  # remove empty
                vocab_counter.update(tokens)

    with open(out_file[:-4] + '_gt.pkl', 'wb') as out:
        pk.dump(data, out)

    print "Finished writing file %s\n" % out_file
    # print 'average extract sents num: ', float(sum(extract_sents_num)) / len(extract_sents_num)
    # print 'average extract words num: ', float(sum(extract_words_num)) / len(extract_words_num)
    # print 'average article sents num: ', float(sum(article_sents_num)) / len(article_sents_num)
    split_name = out_file.split('.')[0]
    extract_info[split_name] = {'extract_sents_num': extract_sents_num,
                                'extract_words_num': extract_words_num,
                                'article_sents_num': article_sents_num}

    # write vocab to file
    if makevocab:
        print "Writing vocab file..."
        with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print "Finished writing vocab file"

if len(sys.argv) == 3 and sys.argv[2] == 'test':
    print "Now in TEST mode"
    lcsts_train_steps = ['test']
    lcsts_valid_counts = [1106]
    file_name_bias = 3

if __name__ == '__main__':
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print "USAGE: python make_datafiles_lcsts.py <LCSTS_ORIGIN_DATA_dir> [test]"
        sys.exit()
    lcsts_data_dir = sys.argv[1]

    # lcsts_data_dir = '/home/yangon/temp/LCSTS_ORIGIN/DATA'
    # print "Now in TEST mode"
    # lcsts_train_steps = ['test']
    # lcsts_valid_counts = [1106]
    # file_name_bias = 3

    if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

    # 对 LCSTS 数据集三部分依次进行处理
    # 其中，会针对 train（PART_I）生成字典（vocab）
    for _count, data_part in enumerate(lcsts_train_steps):
        data_file_name = "PART_%s.txt" % ("I" * (_count + file_name_bias))
        # tokenize_lcsts(os.path.join(lcsts_data_dir, data_file_name), os.path.join(lcsts_data_dir, data_file_name + '.tokenized.pkl'), lcsts_valid_counts[_count])
        write_to_bin_lcsts(data_part, os.path.join(finished_files_dir, data_part + '.bin'))

    # 保存中文与整数词典
    with open(os.path.join(finished_files_dir, 'char2int.pkl'), 'wb') as output_fp:
        pk.dump(char2int, output_fp)
    with open(os.path.join(finished_files_dir, 'int2char.pkl'), 'wb') as output_fp:
        pk.dump(int2char, output_fp)
    
    with open(os.path.join(finished_files_dir, 'extract_info.pkl'), 'wb') as output_file:
        pk.dump(extract_info, output_file)

    # 按照 make_datafiles.py 中 CHUNK_SIZE，分割 train.bin, val.bin, test.bin
    # 并保存到 finished_files_dir/tra_**.bin 等对应文件中
    # chunk_all()