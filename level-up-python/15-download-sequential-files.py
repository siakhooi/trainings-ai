import urllib.request
import os
import re

def download_files(url, output_directory):
    filename=os.path.basename(url)
    match = re.search(r"(.*?)(0*1)(.*)", url)
    if match:
        prefix=match.group(1)
        run_length=len(match.group(2))
        suffix=match.group(3)
    else:
        return

    if not os.path.exists(output_directory):
       os.makedirs(output_directory)
    i=0
    while True:
        i+=1
        url=f"{prefix}{i:0{run_length}}{suffix}"
        filename=os.path.basename(url)
        try:
            # with urllib.request.urlopen(url) as response, open(os.path.join(output_directory, filename), "wb") as output:
            #     data=response.read()
            #     output.write(data)
            #     print(f"Successfully downloaded {filename}")
            urllib.request.urlretrieve(url, os.path.join(output_directory, filename))
            print(f"Successfully downloaded {filename}")
        except urllib.error.HTTPError:
            break

# commands used in solution video for reference
if __name__ == '__main__':
    download_files('http://699340.youcanlearnit.net/image001.jpg', 'images1')
