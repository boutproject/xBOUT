from xbout import open_boutdataset

bd = open_boutdataset().squeeze(drop=True)

bd.bout.animate("n", animate_over="t", x="x", y="z", sep_pos=40)
