def cal_base_ratio(seq):
    AT = (seq.count("A") + seq.count("T")) / (len(seq) * 2)
    AT = '{:.4f}'.format(AT)
    GC = (seq.count("G") + seq.count("C")) / (len(seq) * 2)
    GC = '{:.4f}'.format(GC)
    ref_stat = {
        "A":
        seq.count("A"),
        "T":
        seq.count("T"),
        "G":
        seq.count("G"),
        "C":
        seq.count("C"),
        "AT":
        AT,
        "GC":
        GC,
        "other":
        len(seq) -
        (seq.count("A") + seq.count("T") + seq.count("G") + seq.count("C"))
    }
    return ref_stat
