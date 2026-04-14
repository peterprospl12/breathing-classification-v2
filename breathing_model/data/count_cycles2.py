import os, csv, glob

def count_cycles(label_dir):
    total_exhales = 0
    total_inhales = 0
    total_files = 0
    total_duration_samples = 0
    for f in sorted(glob.glob(os.path.join(label_dir, "*.csv"))):
        total_files += 1
        exhales = 0
        inhales = 0
        max_sample = 0
        with open(f, "r") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            for row in reader:
                if len(row) >= 3:
                    cls = row[0].strip()
                    end = int(row[2].strip())
                    if end > max_sample:
                        max_sample = end
                    if cls == "exhale":
                        exhales += 1
                    if cls == "inhale":
                        inhales += 1
        total_exhales += exhales
        total_inhales += inhales
        total_duration_samples += max_sample
    duration_sec = total_duration_samples / 44100
    return total_files, total_inhales, total_exhales, duration_sec

lines = []

nf, ni, ne, dur = count_cycles("train/label")
lines.append(f"TRAIN: {nf} plikow, {ni} wdechow, {ne} wydechow, {dur:.0f}s = {dur/60:.1f}min")

nf2, ni2, ne2, dur2 = count_cycles("eval_seen_people/label")
lines.append(f"EVAL_SEEN: {nf2} plikow, {ni2} wdechow, {ne2} wydechow, {dur2:.0f}s = {dur2/60:.1f}min")

nf3, ni3, ne3, dur3 = count_cycles("eval_unseen_people/label")
lines.append(f"EVAL_UNSEEN: {nf3} plikow, {ni3} wdechow, {ne3} wydechow, {dur3:.0f}s = {dur3/60:.1f}min")

persons_train = {}
for f in sorted(glob.glob(os.path.join("train/label", "*.csv"))):
    name = os.path.basename(f).lower().split("_")[0]
    exhales = 0
    inhales = 0
    max_sample = 0
    with open(f, "r") as fh:
        reader = csv.reader(fh)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                cls = row[0].strip()
                end = int(row[2].strip())
                if end > max_sample:
                    max_sample = end
                if cls == "exhale":
                    exhales += 1
                if cls == "inhale":
                    inhales += 1
    if name not in persons_train:
        persons_train[name] = {"files": 0, "inhales": 0, "exhales": 0, "samples": 0}
    persons_train[name]["files"] += 1
    persons_train[name]["inhales"] += inhales
    persons_train[name]["exhales"] += exhales
    persons_train[name]["samples"] += max_sample

lines.append("")
lines.append("TRAIN per person:")
for name, d in sorted(persons_train.items()):
    dur_p = d["samples"] / 44100
    lines.append(f"  {name}: {d['files']} plikow, {d['inhales']}inh/{d['exhales']}exh, {dur_p:.0f}s={dur_p/60:.1f}min")

tf = nf+nf2+nf3
ti = ni+ni2+ni3
te = ne+ne2+ne3
td = dur+dur2+dur3
lines.append("")
lines.append(f"TOTAL: {tf} plikow, {ti} wdechow, {te} wydechow, {td:.0f}s = {td/60:.1f}min")

with open("cycle_counts_result.txt", "w") as out:
    out.write("\n".join(lines))
print("\n".join(lines))

