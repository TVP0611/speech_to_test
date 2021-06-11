
text = 'Bật đèn chùm phòng khách'

def action_kw(text):
    kw_act = ["bật", "tắt", "dừng", "đóng", "mở", "ngắt", "on", "off"]
    text = text.lower()
    text1 = text.lower().split(" ")
    for i in text1:
        if i in kw_act:
            text = text.split(i, 1)
            if " " in text:
                text.remove(" ")
            print(text)

action_kw(text)