import sys
import os
import cPickle as pk

char2int =dict()
int2char = list()

def read_dict(dic_dir):
  global int2char
  global char2int
  with open(dic_dir + "/int2char.pkl", 'rb') as fp:
    int2char  = pk.load(fp)

  with open(dic_dir + "/char2int.pkl", "rb") as fp:
    char2int = pk.load(fp)
  
def convert_to_char(int_input):
  ret = list()
  for word_int in int_input:
    if word_int == '.':
      ret.append(".")
      continue
    ret.append(int2char[int(word_int)-1])
  return ret

def red_ref(ref_file):
  ret = list()

def read_input(input_file):
  ret = list()
  with open(input_file, 'r') as fp:
    for line in fp.readlines():
      ret += (line.strip().split())
  return ret

if __name__=='__main__':
  
  ret_dir = sys.argv[1]
  read_dict(sys.argv[2])
  with open('finished.txt' ,'w') as writer:
    for cfile in os.listdir(ret_dir + "/decoded"):
      int_file = os.path.join(ret_dir + "/decoded", cfile)
      ref_file = os.path.join(ret_dir + "/reference", cfile.split("_")[0]+"_reference.txt")
      ref_list = convert_to_char(read_input(ref_file))
      decod_list = convert_to_char(read_input(int_file))
      writer.write(''.join(ref_list) + '::' + ''.join(decod_list) + '\n')
