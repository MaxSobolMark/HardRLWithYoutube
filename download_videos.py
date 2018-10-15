import os
import argparse

def main():
    """ Uses youtube-dl to download entire list of playlists from {filename}
    Format of input file: each line has {youtube_playlist_url} {start_video} {end_video} {*comments}
    """
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--filename', help='environment ID', default='playlists.txt')
    args = args_parser.parse_args()

    playlists_data = open(args.filename, "r")
    for playlist in playlists_data:
        playlist_url, start_video, end_video, *_ = playlist.split()
        os.system('youtube-dl -o videos/{}/%(playlist)s/%(playlist_index)s.%(ext)s --playlist-start {} --playlist-end {} -f "bestvideo[height<=480][ext=mp4]" {}'.format(args.filename, start_video, end_video, playlist_url))

if __name__ == '__main__':
    main()
