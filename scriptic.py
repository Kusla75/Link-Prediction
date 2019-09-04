import os

(dirname, prom) = os.path.split(os.path.dirname(__file__))
folder = os.path.join(dirname, "Link Prediction\\Raw_datasets\\ego-facebook\\")
save_file = os.path.join(dirname, "Link Prediction\\ego-fb_all_featnames")

graphs_ls = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]
f_rows = []

for graph in graphs_ls:
    pom = "{}.featnames".format(graph)
    with open(folder + pom, "r") as f:
        for line in f:
            line = line.split(" ", 1)[1]
            if not line in f_rows:
                f_rows.append(line)

with open(save_file, "w") as sf:
    for row in f_rows:
        sf.write(row)

