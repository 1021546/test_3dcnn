import tables
import tensorflow as tf
import numpy as np

tf.app.flags.DEFINE_string(
    'development_dataset_path', './development_sample_dataset_speaker.hdf5',
    'Directory where checkpoints and event logs are written to.')


# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS

# Load the sample artificial dataset
fileh = tables.open_file(FLAGS.development_dataset_path, mode='r')

# Train
print("Train data shape:", fileh.root.utterance_train.shape)
print(type(fileh.root.utterance_train))
print("Train label shape:", fileh.root.label_train.shape)
print(type(fileh.root.label_train))

# Get the number of subjects
num_subjects = len(np.unique(fileh.root.label_train[:]))
print(num_subjects)

# Test
print("Test data shape:", fileh.root.utterance_test.shape)
print(type(fileh.root.utterance_test))
print("Test label shape:",fileh.root.label_test.shape)
print(type(fileh.root.label_test))

# Get the number of subjects
num_subjects = len(np.unique(fileh.root.label_test[:]))
print(num_subjects)


# tf.app.flags.DEFINE_string(
#     'enrollment_dataset_path', './enrollment-evaluation_sample_dataset.hdf5',
#     'Directory where checkpoints and event logs are written to.')


# # Store all elemnts in FLAG structure!
# FLAGS = tf.app.flags.FLAGS

# # Load the artificial datasets.
# fileh = tables.open_file(FLAGS.enrollment_dataset_path, mode='r')

# # Train
# print("Enrollment data shape:", fileh.root.utterance_enrollment.shape)
# print("Enrollment label shape:", fileh.root.label_enrollment.shape)

# # Get the number of subjects
# num_subjects = len(np.unique(fileh.root.label_enrollment[:]))
# print(num_subjects)

# # Test
# print("Evaluation data shape:", fileh.root.utterance_evaluation.shape)
# print("Evaluation label shape:",fileh.root.label_evaluation.shape)

# # Get the number of subjects
# num_subjects = len(np.unique(fileh.root.label_evaluation[:]))
# print(num_subjects)



# Result

# development_sample_dataset_speaker.hdf5
# Train data shape: (12, 80, 40, 20)
# <class 'tables.earray.EArray'>
# Train label shape: (12,)
# <class 'tables.earray.EArray'>
# 4
# Test data shape: (12, 80, 40, 20)
# <class 'tables.earray.EArray'>
# Test label shape: (12,)
# <class 'tables.earray.EArray'>
# 4
# Closing remaining open files:./development_sample_dataset_speaker.hdf5...done


# enrollment-evaluation_sample_dataset.hdf5
# Enrollment data shape: (108, 80, 40, 1)
# Enrollment label shape: (108,)
# 4
# Evaluation data shape: (12, 80, 40, 1)
# Evaluation label shape: (12,)
# 4
# Closing remaining open files:./enrollment-evaluation_sample_dataset.hdf5...done