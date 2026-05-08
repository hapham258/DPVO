import pickle
from pprint import pprint

pkl_path = "datasets/TartanAir.pickle"

with open(pkl_path, "rb") as f:
    obj = pickle.load(f)

print("Top-level type:", type(obj))

if isinstance(obj, (list, tuple)):
    print("Top-level length:", len(obj))
    for i, item in enumerate(obj[:5]):
        print(f"\n--- top-level[{i}] ---")
        print("type:", type(item))
        if hasattr(item, "__len__"):
            try:
                print("len:", len(item))
            except Exception:
                pass

# Your code uses [0]
scene_info = obj[0]
print("\nscene_info type:", type(scene_info))

if isinstance(scene_info, dict):
    print("Number of scenes:", len(scene_info))

    # first few scene names
    scene_names = list(scene_info.keys())[:3]
    print("\nExample scene names:")
    pprint(scene_names)

    # inspect one scene
    first_scene = scene_names[0]
    print(f"\nInspecting scene: {first_scene}")

    entry = scene_info[first_scene]
    print("Entry type:", type(entry))

    if isinstance(entry, dict):
        print("Keys:")
        pprint(list(entry.keys()))

        # inspect values shallowly
        for k, v in entry.items():
            print(f"\nKey: {k}")
            print("  type:", type(v))
            if isinstance(v, (list, tuple)):
                print("  len:", len(v))
                if len(v) > 0:
                    print("  first item type:", type(v[0]))
                    print("  first item:", v[0])
            elif isinstance(v, dict):
                print("  dict keys:", list(v.keys())[:10])
            else:
                print("  value:", v)

elif isinstance(scene_info, list):
    print("scene_info length:", len(scene_info))
    print("First element type:", type(scene_info[0]))
    print("First element:")
    pprint(scene_info[0])

