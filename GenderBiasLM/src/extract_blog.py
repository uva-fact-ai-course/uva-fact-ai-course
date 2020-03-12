
import zipfile
import os
import shutil
import requests


def substring_indexes(substring, string):
    """
    Generate indices of where substring begins in string

    >>> list(find_substring('me', "The cat says meow, meow"))
    [13, 19]
    """
    last_found = -1  # Begin at -1 so the next position to search from is 0
    while True:
        # Find next index of substring, by starting after its last known position
        last_found = string.find(substring, last_found + 1)
        if last_found == -1:
            break  # All occurrences have been found
        yield last_found


def download_blogs(zip_filename):
    url = "http://www.cs.biu.ac.il/~koppel/blogs/blogs.zip"
    r = requests.get(url)

    with open(zip_filename, 'wb') as f:
        f.write(r.content)
    f.close()


def extract_blogs(zip_filename):

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall('.')


def get_posts_from_file(file):
    print(file)

    filedata = " ".join([line.decode(errors="replace").replace("\n", "") for line in open(file, "rb").readlines()])
    start_indices = [index+6 for index in substring_indexes("<post>",filedata)]
    end_indices = [index for index in substring_indexes("</post>",filedata)]
    return [filedata[start_indices[i]:end_indices[i]] for i in range(len(start_indices))]


def save_formatted_blogpost(write_filename, text):
    with open(write_filename, "w+") as f:
        f.write(text)
    f.close()




def convert_blogposts():

    target_directory = "extracted_blogs"
    if not os.path.exists(target_directory):
        os.mkdir(target_directory)


    for root, dirs, files in os.walk(os.path.join('.', 'blogs')):
        n_files = len(files)
        for i, file in enumerate(files):
            print(f"converting xml of blog {i}/{n_files} to usable txt")
            blogfile = os.path.join(root, file)
            blog_text = " ".join(get_posts_from_file(blogfile))

            new_file = os.path.join(target_directory, file[:-4] + ".txt")
            save_formatted_blogpost(new_file, blog_text)



def replace_blogdir():
    shutil.rmtree("blogs")
    shutil.move("extracted_blogs", "blogs")



if __name__ == "__main__":

    final_dirname = "blogs"
    zip_filename = "blogs.zip"

    if not os.path.exists(zip_filename):
        print("downloading dataset...")
        download_blogs(zip_filename)

    if not os.path.exists(final_dirname):
        print("unzipping...")
        extract_blogs()

    print("converting xml to traindata...")
    convert_blogposts()
    replace_blogdir()
