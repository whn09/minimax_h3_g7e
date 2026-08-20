# 所有 fl2va / ref2va 脚本共用的 prompt。用法：先设好 $IMG，再 `. assets/prompts.sh`，
# 然后拿 $FL2VA_PROMPT / $REF2VA_PROMPT。
#
# **prompt 必须跟输入图配对**，这里按 $IMG 的文件名选：
#   input_cat.jpg  白天窗台白猫照（1200x811，**不在这个库里**，放本地素材目录 / 机器上）。历史上
#                  的性能表都是用它量的 —— 要跟旧数字对齐就用它。
#   first.png      公开样例（= assets/ref.mp4 第 0 帧，864x480 的**夜景卧室**：暖色台灯、床上有人
#                  睡着、地毯上三只小猫在打闹）。库里默认它，是为了脚本对外能直接跑。
#
# 为什么单独一个文件：白猫 prompt 被六个脚本各抄了一份，而库里的 IMG 默认是 first.png —— 于是在
# 没有那张白猫图的机器上，跑出来的是"从夜景卧室强行漂到午后白猫"的片子：小猫消失、窗外天亮、窗台上长出
# 一只白猫，看着像"从黑慢慢变白"。整轮 Cache-DiT RDT 扫描就是这么跑掉的（RDT 是帧间残差阈值，
# 而这种全局重打光把残差顶得极高，gate 的行为跟正常片不是一回事，那批画质数只能作废）。
#
# 加素材时：新图 -> 在下面 case 里加一支，别在调用方脚本里写 prompt 字面量。

case "$(basename "${IMG:-assets/first.png}")" in
  input_cat.*)
    FL2VA_PROMPT=${FL2VA_PROMPT:-"A white cat sitting on an open window ledge slowly turns its head toward the camera, blinks, and gently lifts one paw while a soft breeze moves the curtains. Natural afternoon light, subtle street ambience and soft paw sounds, realistic motion, static cinematic camera."}
    REF2VA_PROMPT=${REF2VA_PROMPT:-"Use <Picture 1> as the visual subject. The same white cat sits beside an open window, slowly turns toward the camera, blinks, and gently lifts one paw while a soft breeze moves the curtains. Preserve the cat's identity and markings, natural afternoon light, realistic coherent motion, synchronized soft ambience, static cinematic camera."}
    ;;
  *)
    # first.png 及其它：描述那张夜景图本身，动作从那一帧自然往下走。
    FL2VA_PROMPT=${FL2VA_PROMPT:-"A dim bedroom at night. Three kittens keep tumbling and batting at each other on the patterned rug in front of the radiator, tails flicking, while the person in the bed stays asleep under the blanket. The bedside lamp keeps its steady warm glow, the tall curtains stir faintly, night sky outside the windows. Warm low-key lamp light, static cinematic camera, quiet room tone with soft paw pats and faint fabric rustle."}
    REF2VA_PROMPT=${REF2VA_PROMPT:-"Use <Picture 1> as the visual subject. The same dim night bedroom: the three kittens go on tussling on the rug by the radiator, the sleeping person does not stir, the bedside lamp holds its warm glow and the curtains move slightly. Preserve the layout, colours and low-key lighting of the reference, realistic coherent motion, static cinematic camera, synchronized quiet ambience."}
    ;;
esac
# 跑错输入是静默的（图不在机器上就落到默认样例图），所以每次都把配对打进日志。
echo "PROMPT_PAIR img=${IMG:-assets/first.png}"
