"""
sumarize.py
"""
import os

def write_summary_data():
    with open('summary.txt','w') as summary:
        if (os.path.exists('fetched_data/cluster_stats.txt')):
            file_open = open('fetched_data/cluster_stats.txt','r')
            for name in file_open:
                summary.write(name)
            summary.write('\n')
            file_open.close()
        if (os.path.exists('fetched_data/classify_stats.txt')):
            file_open = open('fetched_data/classify_stats.txt','r')
            for name in file_open:
                summary.write(name)
            summary.write('\n')
            file_open.close()
    summary.close()

    
def main():
    write_summary_data()

if __name__ == '__main__':
    main()