import cv2
import numpy as np
import argparse
import os
from VAEFeaturizer import VAEFeaturizer
from TDCFeaturizer import TDCFeaturizer
from ForwardModelFeaturizer import ForwardModelFeaturizer

def train_featurizer(featurizer_type, videos_path, framerate, initial_width, initial_height,
                     desired_width, desired_height, epochs, batch_size, learning_rate,
                     featurizer_save_path=None, restore_model=False):
    """Preprocesses videos to create dataset, and trains a featurizer

    Converts the videos of every playlist in a given directory to monochromatic,
    to a certain size, and framerate.

    Args:
        featurizer_type: One of 'tdc', 'vae', or 'forward_model'
        videos_path: Inside the 'videos/' directory, the name of the subdirectory for videos.
        framerate: The desired framerate of the dataset.
        initial_width: The videos will be resized to this width.
        initial_height: The videos will be resized to this height.
        desired_width: The agent randomly crops the video to this width.
        desired_height: The agent randomly crops the video to this width.
        epochs: Times we should run the training step.
        batch_size: The batch size to use for training and inference.
        learning_rate: learning rate of the training algorithm.
        featurizer_save_path: path to save the parameters of the featurizer when it's done training.
        restore_model: whether we should load a pre-trained model; If true, load from featurizer_save_path.

    Returns:
        Instance of trained featurizer.

    Raises:
        TypeError: An unknown featurizer type was given.

    """
    # Prepare dataset
    dataset = generate_dataset(videos_path, framerate, initial_width, initial_height)


    if featurizer_type == 'vae':
        featurizer_class = VAEFeaturizer
    elif featurizer_type == 'tdc':
        featurizer_class = TDCFeaturizer
    elif featurizer_type == 'forward_model':
        featurizer_class = ForwardModelFeaturizer
    else:
        raise TypeError
    
    featurizer = featurizer_class(initial_width, initial_height, desired_width, desired_height, feature_vector_size=1024, learning_rate=learning_rate, experiment_name='default')

    if restore_model:
        featurizer.load(featurizer_save_path)

    featurizer.train(dataset, epochs, batch_size)

    if featurizer_save_path:
        featurizer.save(featurizer_save_path)

    return featurizer

def generate_dataset(videos_path, framerate, width, height):
    """Converts videos from specified path to ndarrays of shape [numberOfVideos, -1, width, height, 1]

    Args:
        videos_path: Inside the 'videos/' directory, the name of the subdirectory for videos.
        framerate: The desired framerate of the dataset.
        width: The width we will resize the videos to.
        height: The height we will resize the videos to.

    Returns:
        The dataset with the new size and framerate, and converted to monochromatic.

    """
    dataset = []
    video_index = 0
    for playlist in os.listdir('videos/' + videos_path):
        for video_name in os.listdir('videos/{}/{}'.format(videos_path, playlist)):
            dataset.append([])
            print('Video: {}'.format(video_name))
            video = cv2.VideoCapture('videos/{}/{}/{}'.format(videos_path, playlist, video_name))
            while video.isOpened():
                success, frame = video.read()
                if success:
                    frame = preprocess_image(frame, width, height)
                    dataset[video_index].append(frame)

                    frame_index = video.get(cv2.CAP_PROP_POS_FRAMES)
                    video_framerate = video.get(cv2.CAP_PROP_FPS)
                    video.set(cv2.CAP_PROP_POS_FRAMES, frame_index + video_framerate // framerate)
                    last_frame_index = video.get(cv2.CAP_PROP_FRAME_COUNT)
                    if frame_index >= last_frame_index:
                        # Video is over
                        break

                    
                else:
                    break
            dataset[video_index] = np.reshape(dataset[video_index], (-1, width, height, 1))
            video_index += 1
    return dataset

def preprocess_image(image, width, height):
    """ Changes size, makes image monochromatic """

    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array(image, dtype=np.uint8)
    image = np.expand_dims(image, -1)
    return image

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--featurizer_type', help='Choose from [tdc, vae, forward_model]', default='tdc')
    args_parser.add_argument('--videos_path', help='Path for training data')
    args_parser.add_argument('--featurizer_save_path', help='Path for saving the featurizer', default='default')
    args_parser.add_argument('--framerate', help='Desired FPS for the videos', type=float, default=0.25)
    args_parser.add_argument('--initial_width', help='Initial width for the videos', type=int, default=140)
    args_parser.add_argument('--initial_height', help='Initial height for the videos', type=int, default=140)
    args_parser.add_argument('--desired_width', help='Width for the videos after cropping', type=int, default=128)
    args_parser.add_argument('--desired_height', help='Height for the videos after cropping', type=int, default=128)
    args_parser.add_argument('--num_epochs', help='Number of epochs for training', type=int, default=10)
    args_parser.add_argument('--batch_size', help='Batch size for training', type=int, default=32)
    args_parser.add_argument('--learning_rate', help='Learning rate for training', type=float, default=0.0001)
    args_parser.add_argument('--restore_model', help='Start training from a saved checkpoint (featyruzer_save_path)', type=bool, default=False)
    args = args_parser.parse_args()
    train_featurizer(
        args.featurizer_type,
        args.videos_path,
        args.framerate,
        args.initial_width,
        args.initial_height,
        args.desired_width,
        args.desired_height,
        args.num_epochs,
        args.batch_size,
        args.learning_rate,
        args.featurizer_save_path,
        args.restore_model,
    )
