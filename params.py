from funcData import *
import funcCustom
import numpy as np

MOVING_AVERAGE_DECAY = 0.999
WEIGHT_DECAY_FACTOR = 0.0001
daystr,timestr=funcCustom.beautifultime()
FLAGS = tf.app.flags.FLAGS

# Basic model parameters which will be often modified.
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('num_epochs', 200,
                            """Number of epochs to train. -1 for unlimited""")
#for quantization network
LR = tf.Variable(initial_value=0., trainable=False, name='lr', dtype=tf.float32)
tf.app.flags.DEFINE_string("LR_schedule","[0, 3,15,2,30, 1,200,0]"
                           ,"Please input the list which have learning rate schedule")

#for conventional network
tf.app.flags.DEFINE_float('learning_rate', 0.0001,
                            """Initial learning rate used.""")
tf.app.flags.DEFINE_string('dataset', 'MNIST_rand',
                           """Name of dataset used.""")
tf.app.flags.DEFINE_string('model','MLP_00_Basic_3',    #'Probe_MLP',MLP_00_Basicvanila  MLP_00_Basic_512*3
                           """Name of loaded model.""")
tf.app.flags.DEFINE_integer('patience',0,'# of patience')
tf.app.flags.DEFINE_integer('seed',333,"num of random seed")
# Level2 Parameters which will be sometimes modified.
# 2-1 Quantization options
tf.app.flags.DEFINE_integer('W_real_target_level', 128, """Target level.""")
tf.app.flags.DEFINE_integer('W_target_level', 9999, """Target level.""")
tf.app.flags.DEFINE_integer('Wq_target_level',  8, """Target level.""")
tf.app.flags.DEFINE_integer("num_cell",1,"The number of the cells we use per weight")
tf.app.flags.DEFINE_string('level_info_name',"level_info_mygenerate.xlsx","level_info file name")
tf.app.flags.DEFINE_integer('choice', 0, """select which type to choose""")    # which writing voltage set to choose

# 2-2 Properties : Variation and drift
tf.app.flags.DEFINE_integer("writenorm",2,"""Temporary""")
tf.app.flags.DEFINE_string('Inter_variation_options', "True,False,1,True,0.05"
                           , """Include Cell-to-cell(Inter-cell) variation or not.""")
tf.app.flags.DEFINE_string('Intra_variation_options', "False,False,1,True,1"
                           , """In-cell(Intra-cell) variation or not""")
# 1. Use variation or not
# 2. if use variation, use random number or true param   3. if use random number, write a targeted variation
# 4. if use variation, adjust variation or not           5. if adjust variation, write a coefficient using in adjusting
tf.app.flags.DEFINE_bool('Drift1', False, """Drift or Not.""")
tf.app.flags.DEFINE_bool('Drift2', False, """Drift or Not.""")

# 2-3 Logging data
tf.app.flags.DEFINE_bool('summary', True, """Record summary.""")   #Log only include accuracy data
tf.app.flags.DEFINE_bool('summary2', False, """Record summary2.""")#Log also include histogram, weight's scalar graph..etc
tf.app.flags.DEFINE_integer('row_choice', 0, """select which type to choose""") # which row of the excel file to write in the end
tf.app.flags.DEFINE_integer('col_choice', 1, """select which type to choose""") # which column of the excel file to write in the end
tf.app.flags.DEFINE_string('memo',"set1","""select which type to choose""")

# Level3 Parameters which will be seldom modified.
# 3-1 Path options
tf.app.flags.DEFINE_string('daystr', daystr,"""Name of saved dir.""")
Inter=eval('['+FLAGS.Inter_variation_options+']')
Intra=eval('['+FLAGS.Intra_variation_options+']')
if (Inter[0]==True) and (Intra[0]==True):
    part_name="4_(Var"
    if (Intra[3] == True):
        part_name += "_InCAdj="+str(Intra[4])
    if (Inter[3] == True):
        part_name += "_CtoCAdj="+str(Inter[4])
elif (Inter[0]==True) and (Intra[0]==False):
    part_name="3_(Cell-to-cell"
    if (Inter[3] == True):
        part_name += "_Adj="+str(Inter[4])
elif (Inter[0]==False) and (Intra[0]==True):
    part_name="2_(In-cell"
    if (Intra[3] == True):
        part_name += "_Adj="+str(Intra[4])
elif (Inter[0]==False) and (Intra[0]==False):
    part_name="1_(Vanila"
else:
    exit()
part_name+=")"
LR_schedule = eval(FLAGS.LR_schedule)
bits = np.ceil(np.log(FLAGS.W_real_target_level) / np.log(2.0))
write_read=(bits<12) and (Inter[0] or Intra[0])
num_level = int(0.999999+np.exp(np.log(FLAGS.W_real_target_level) / FLAGS.num_cell)) if write_read else FLAGS.W_real_target_level
FLAGS.W_target_level = num_level
tf.app.flags.DEFINE_string('checkpoint_dir', './results/'+daystr+'/'+part_name+'('+FLAGS.model+'-'+FLAGS.dataset+')_'
                           +"("+str(FLAGS.W_real_target_level)+','+str(FLAGS.W_target_level)+','+str(FLAGS.Wq_target_level)+")"+'levels_'
                           +str(FLAGS.num_cell)+"cells_"+"(time="+timestr+")"+"(memo="+FLAGS.memo+")",
                           """Constant""")
tf.app.flags.DEFINE_string('log_dir', FLAGS.checkpoint_dir + '/log/',"""Constant""")
# 3-2 선택적으로 쓰는 옵션:
tf.app.flags.DEFINE_bool('bit_deterministic', False,       # -1,1,1.4로 양자화 함(2비트)
                           """2bit_deterministic or Not.""")
tf.app.flags.DEFINE_bool('Weight_decay', False,
                           """Weightdecay or Not.""")
tf.app.flags.DEFINE_bool('Fine_tuning', False,            #FC layer만 작동시킴
                           """Fine_tuning or Not.""")
tf.app.flags.DEFINE_string('load_dir', '',"""Name of loaded dir.""")
tf.app.flags.DEFINE_bool('Load_checkpoint', False,        #체크포인트를 불러 올 것인가?
                           """Load_checkpoint or Not.""")
load_checkpoint_path='./results/today/05cifar10_BNN_big2(Drift1_Var_125epoch_81%)'
# tf.app.flags.DEFINE_string('log', 'ERROR',
#                            'The threshold for what messages will be logged '
#                             """DEBUG, INFO, WARN, ERROR, or FATAL.""")

tf.set_random_seed(FLAGS.seed)
if FLAGS.dataset=="MNIST_back":
    # dataset_filename_train="./Datasets/MNIST_back/mnist_background_images_train.amat"
    # dataset_filename_test = "./Datasets/MNIST_back/mnist_background_images_test.amat"
    # images_train, labels_train = funcCustom.load_dataset_as_numpy(dataset_filename_train)
    # images_test, labels_test = funcCustom.load_dataset_as_numpy(dataset_filename_test)
    # np.save("MNIST_back_train_images.npy", images_train)
    # np.save("MNIST_back_test_images.npy", images_test)
    # np.save("MNIST_back_train_labels.npy", labels_train)
    # np.save("MNIST_back_test_labels.npy", labels_test)
    images_train =np.load("./Datasets/MNIST_back/MNIST_back_train_images.npy")

    images_test = np.load("./Datasets/MNIST_back/MNIST_back_test_images.npy")
    labels_train =np.load("./Datasets/MNIST_back/MNIST_back_train_labels.npy")
    labels_test = np.load("./Datasets/MNIST_back/MNIST_back_test_labels.npy")
elif FLAGS.dataset== "MNIST_rand":
    # dataset_filename_train="./Datasets/MNIST_backrand/mnist_background_random_train.amat"
    # dataset_filename_test = "./Datasets/MNIST_backrand/mnist_background_random_test.amat"
    # images_train, labels_train = funcCustom.load_dataset_as_numpy(dataset_filename_train)
    # images_test, labels_test = funcCustom.load_dataset_as_numpy(dataset_filename_test)
    # np.save("MNIST_rand_train_images.npy", images_train)
    # np.save("MNIST_rand_test_images.npy", images_test)
    # np.save("MNIST_rand_train_labels.npy", labels_train)
    # np.save("MNIST_rand_test_labels.npy", labels_test)
    images_train =np.load("./Datasets/MNIST_backrand/MNIST_rand_train_images.npy")
    images_test = np.load("./Datasets/MNIST_backrand/MNIST_rand_test_images.npy")
    labels_train =np.load("./Datasets/MNIST_backrand/MNIST_rand_train_labels.npy")
    labels_test = np.load("./Datasets/MNIST_backrand/MNIST_rand_test_labels.npy")



