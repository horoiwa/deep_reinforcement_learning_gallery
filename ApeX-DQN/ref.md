100×100=10000遷移を集める
- 2actor 34.0 sec
- 4actor 22.5 sec
- 6actor 17.4 sec
- 8actor 15.2 sec

- 23 actor 6sec


100遷移の格納
- 0.01-0.05 sec

16バッチ作成(from 10000 transition)
- 0.15 - 0.2 sec

16バッチ転送のみ(from 10000 transition)
- 0.2 sec

16バッチ解凍(from 10000 transition)
- 4-5 sec #:16バッチのexp解凍に4秒
直列で処理すると１バッチの解凍だけで0.4-0.5秒程度消費する

16バッチthreading解凍(from 10000 transition)
- 2.5 sec
threadingの数を増やしすぎると最初のバッチがreadyになるのが遅くなるのでパフォーマンスが下がる

16バッチ解凍, 勾配計算(from 10000 transition)
- 14-16 sec #: 勾配計算一回で2秒くらい？

16バッチ threading 解凍, 勾配計算(from 10000 transition)
- 5 sec

16バッチ優先度更新：
0.1-0.15 sec
