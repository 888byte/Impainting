import re
import numpy as np

lux_pat = re.compile(r'^Lux:(\d+)$')
temp_pat = re.compile(r'^(\d{2})Temperature:\s*([0-9.]+)\s*C\s*Humidity:\s*([0-9.]+)\s*%RH$')
no_pat = re.compile(r'^NO\.(\d+)\s+R:(\d+)\s+G:(\d+)\s+B:(\d+)$')

def parse_txt(path: str):
    with open(path, "rb") as f:
        s = f.read().decode("utf-8", errors="ignore")
    lines = [ln.strip() for ln in s.replace("\r", "\n").split("\n") if ln.strip()]

    rows = []
    lux_buffer = []
    cur = None

    def finalize():
        nonlocal cur
        if cur is None:
            return
        for i in range(1, 10):
            for c in ["R","G","B"]:
                cur.setdefault(f"NO{i}_{c}", np.nan)
        rows.append(cur)
        cur = None

    for ln in lines:
        if ln.startswith("Testing_Left"):
            continue

        m = lux_pat.match(ln)
        if m:
            lux_buffer.append(float(m.group(1)))
            lux_buffer = lux_buffer[-3:]
            continue

        m = temp_pat.match(ln)
        if m:
            finalize()
            temp = float(m.group(2))
            hum = float(m.group(3))

            lux_vals = [np.nan]*3
            last = lux_buffer[-3:]
            for i, v in enumerate(last):
                lux_vals[3-len(last)+i] = v

            cur = {
                "temperature": temp,
                "humidity": hum,
                "lux1": lux_vals[0],
                "lux2": lux_vals[1],
                "lux3": lux_vals[2],
            }
            continue

        m = no_pat.match(ln)
        if m and cur is not None:
            idx = int(m.group(1))
            cur[f"NO{idx}_R"] = float(m.group(2))
            cur[f"NO{idx}_G"] = float(m.group(3))
            cur[f"NO{idx}_B"] = float(m.group(4))

    finalize()
    return rows

def build_series(rows):
    # 输出: [9, T, 8]  8维 = RGB(3)+temp/hum(2)+lux(3)
    T = len(rows)
    out = np.zeros((9, T, 8), dtype=np.float32)
    for t, r in enumerate(rows):
        env = np.array([r["temperature"], r["humidity"], r["lux1"], r["lux2"], r["lux3"]], dtype=np.float32)
        for i in range(1, 10):
            rgb = np.array([r[f"NO{i}_R"], r[f"NO{i}_G"], r[f"NO{i}_B"]], dtype=np.float32)
            out[i-1, t, :3] = rgb
            out[i-1, t, 3:] = env
    return out

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_txt", required=True)
    ap.add_argument("--out", required=True, help="e.g. data/color_series.npy")
    args = ap.parse_args()

    rows = parse_txt(args.in_txt)
    series = build_series(rows)
    np.save(args.out, series)
    print("saved:", args.out, "shape:", series.shape)
