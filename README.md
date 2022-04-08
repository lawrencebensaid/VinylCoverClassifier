# Vinyl Cover Classifier

Highlight all albums in the wall

## Run

```bash
$ python imageMatching.py --original-dir data/original --test-img data/test/Canvas.jpg
```

## Demo

`result.jpg`

<img src="https://user-images.githubusercontent.com/43364935/162458304-2834bedc-6f7e-4a63-b97c-7f6b53ac3899.jpg" width="50%">

`results.json`
```json
[
  {
    "name": "LanaDelray",
    "confidence": 0.64,
    "points": [
      [2373, 265],
      [2359, 869],
      [2944, 876],
      [2961, 279]
    ]
  },
  {
    "name": "AlexTurner",
    "confidence": 0.38,
    "points": [
      [137, 1614],
      [223, 2129],
      [653, 2112],
      [677, 1637]
    ]
  },
  {
    "name": "GunsNRoses",
    "confidence": 0.72,
    "points": [
      [2357, 979],
      [2347, 1559],
      [2919, 1565],
      [2942, 982]
    ]
  },
  {
    "name": "Bowie",
    "confidence": 0.76,
    "points": [
      [890, 242],
      [907, 851],
      [1500, 858],
      [1499, 249]
    ]
  },
  {
    "name": "ArcticMonkeys",
    "confidence": 0.3,
    "points": [
      [1350, 1666],
      [1349, 2130],
      [1817, 2124],
      [1823, 1657]
    ]
  },
  {
    "name": "GeorgeMichael",
    "confidence": 0.54,
    "points": [
      [3170, 405],
      [3154, 1012],
      [3720, 1017],
      [3753, 425]
    ]
  },
  {
    "name": "Froukje",
    "confidence": 0.5,
    "points": [
      [2437, 1642],
      [2467, 2090],
      [2888, 2097],
      [2956, 1639]
    ]
  },
  {
    "name": "KillersHotFuss",
    "confidence": 0.46,
    "points": [
      [3070, 989],
      [3050, 1562],
      [3607, 1563],
      [3635, 994]
    ]
  },
  {
    "name": "Boney",
    "confidence": 0.74,
    "points": [
      [1641, 972],
      [1640, 1553],
      [2215, 1557],
      [2225, 982]
    ]
  },
  {
    "name": "NirvanaHormoaning",
    "confidence": 0.54,
    "points": [
      [1934, 1681],
      [1914, 2092],
      [2336, 2095],
      [2349, 1693]
    ]
  },
  {
    "name": "Elton",
    "confidence": 0.94,
    "points": [
      [900, 966],
      [913, 1557],
      [1508, 1556],
      [1505, 973]
    ]
  },
  {
    "name": "BeatlesMystery",
    "confidence": 0.6,
    "points": [
      [766, 1670],
      [778, 2135],
      [1248, 2129],
      [1246, 1663]
    ]
  },
  {
    "name": "HarryStyles",
    "confidence": 0.52,
    "points": [
      [1632, 263],
      [1642, 859],
      [2240, 864],
      [2246, 254]
    ]
  },
  {
    "name": "Beatles",
    "confidence": 0.86,
    "points": [
      [161, 969],
      [184, 1561],
      [777, 1559],
      [764, 967]
    ]
  },
  {
    "name": "elo",
    "confidence": 0.56,
    "points": [
      [128, 231],
      [156, 846],
      [765, 858],
      [750, 242]
    ]
  }
]
```
