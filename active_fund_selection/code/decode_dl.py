"""Decode Drive download tool-result JSON files into dl/<title>/<title>."""
import base64, json, sys, os, re
out_root = sys.argv[1]
for p in sys.argv[2:]:
    d = json.load(open(p))
    title = d['title']
    safe = re.sub(r'[^\w.\-一-鿿]', '_', title)
    od = os.path.join(out_root, safe.split('.')[0])
    os.makedirs(od, exist_ok=True)
    raw = base64.b64decode(d['content'])
    fp = os.path.join(od, safe)
    open(fp, 'wb').write(raw)
    print(f'{title}\t{len(raw):,}\t{fp}')
