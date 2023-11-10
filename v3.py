# -*- coding:utf-8 -*-
import json
import os
import re
import sys
import threading
import urllib

import jieba
import pandas
import requests
import wordcloud
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton, QLabel, QLineEdit, QVBoxLayout

from BERTuse import bertAnalyze


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.InitUI()

    def InitUI(self):
        self.urlEdit = QLineEdit(self)
        self.urlEdit.move(100, 100)
        self.urlEdit.resize(400, 30)
        self.startBtn = QPushButton('开始爬取', self)
        self.startBtn.move(250, 150)
        self.pictureLabel = QLabel(self)
        self.pictureLabel.move(0, 160)
        self.startBtn.clicked.connect(self.main)
        self.resize(600, 800)

    def main(self):
        # 浏览器访问伪装
        headers = {
            'cookie': '__jdu=1384393634; user-key=64cfaaaa-e9f5-44b6-aac1-dec3f039c579; shshshfpa=40046db0-abc7-9249-e586-be928ab56582-1622550747; shshshfpb=dZkSO4sODsTa9GxzwSxHzdg%3D%3D; pinId=jHv0wxmPkPhb-YWw04wmnw; pin=jd_YJrEMvrufaok; unick=YWH%E6%B5%A9%E4%BB%94; _tp=DEtfmumI9fa5on%2Bb%2Fy6SRQ%3D%3D; _pst=jd_YJrEMvrufaok; areaId=17; unpl=V2_ZzNtbUMHSxRyARJRfEkJDGIAEV9KBEEXcVoVBnlOCwFjURRfclRCFnUUR1NnGl0UZwsZWEtcQh1FCEdkeBBVAWMDE1VGZxBFLV0CFSNGF1wjU00zQwBBQHcJFF0uSgwDYgcaDhFTQEJ2XBVQL0oMDDdRFAhyZ0AVRQhHZHsbVQBlCxBaQFJzJXI4dmR%2bEFQDYwYiXHJWc1chVEFTexlVBSoDEFRHVUsXcgpDZHopXw%3d%3d; __jdv=76161171|baidu-pinzhuan|t_288551095_baidupinzhuan|cpc|0f3d30c8dba7459bb52f2eb5eba8ac7d_0_0a9068e46ad842239b335cbc3ff55c73|1624189690392; __jda=122270672.1384393634.1622550742.1624005988.1624189690.6; __jdc=122270672; mt_xid=V2_52007VwMQW1haWlgYSxxsUmJTRVtcUVdGFh4bCxliARpUQVFQUxtVH1wCZAIaW1tZAg8deRpdBmcfE1VBW1NLH0ESWARsAhpiX2hRahxIH1QAYjMQWl9Y; shshshfp=d971a9ca9226648ae35be91e17883476; ipLoc-djd=17-1381-50712-50817; wlfstk_smdl=fwobglgnpuy0fctfigmtt2kc0z93efpu; TrackID=1jROvkCTpXm2DN_wcuc8AjFyluDo53jyt5RYqAYS6OTFnr5qXgMhE4-Fvpf7HpcnAswDh7QRz8COJ2FoB3mNhsnTrzqOznZf5TrXM8U-K-_SbcAcnrG1kOspC80SZyE2o; thor=9D33544FC7BC50DC3BFF5A02A674C4D4C72329F4972D9460CD5979A28472933D0E495273852F4E5850AF64D3A4CE7EF313064D74BF8EC4C16BBB49AC07427937FA31D2A5511EE6E0E81386B5F1D42081D4C9BB485DC25641F703E87379C636BF1E3B451B7945BAF969425A8F73BA45D177E7A3B810E864C7402324424CB5D74F840162F4982469A91A77C4DA2DC57B32884CF530A32F32471FDE457D63157D93; ceshi3.com=201; __jdb=122270672.8.1384393634|6.1624189690; shshshsID=92eeac190750957144edad7fd8c96ffd_5_1624190862853; 3AB9D23F7A4B3C9B=NTPTDLYF3NV376QRU35AFKBB3TG5RKSHLDCAPU5ZLXHE53ZIYOIYMRSOZT3DZ2JPFYGUH44G6T2UQZCFXEQA6OAY4M',
            'referer': 'https://item.jd.com/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.77 Safari/537.36 Edg/91.0.864.37',
        }
        url_input = self.urlEdit.text()
        try:
            result = []  # 筛选过后的评论
            reason = []  # 统计每个被筛选淘汰的评论的淘汰理由
            id = ''.join(re.findall(r'com/(.*?)\.html', url_input))
            url = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=' + str(
                id) + '&score=0&sortType=5&page=0&pageSize=10&isShadowSku=0&fold=1'
            print(url)
            data = requests.get(url, headers=headers).content.decode('gbk')
            data = data[data.find('{'):data.rfind('}') + 1]
            data = json.loads(data)
            lastpage = data['maxPage']
            print('共有' + str(lastpage) + '页评论')
            threads = []
            threads.append(
                threading.Thread(target=get_information, args=(id, headers, 0, lastpage // 2, result, reason)))
            threads.append(
                threading.Thread(target=get_information, args=(id, headers, lastpage // 2, lastpage, result, reason)))
            # 开启新线程
            for t in threads:
                t.start()
            # 等待所有线程完成
            for t in threads:
                t.join()
            print("退出主线程")
        except:
            print("提取商品网址信息出错，请重新输入京东商品网址：")
            self.main()
        # 创建新线程,分2个线程（太少了速度慢，太快了容易被封）
        # 将筛选后的评论装入文件
        print("有效评论已装入NEW评论.xlsx文件")
        print("淘汰评论已装入淘汰评论.xlsx文件")
        NEW = pandas.DataFrame(result)
        NEW.to_excel(r'.\NEW评论.xlsx')
        Eliminate = pandas.DataFrame(reason)
        Eliminate.to_excel(r'.\淘汰评论.xlsx')
        comment_word = ''
        analyze = []
        for _comment in result:
            comment_word += _comment["评论内容"]
            analyze.append(_comment["评论内容"])
        all_word = comment_word
        cloud_word(comment_word, 'comment_word')
        print("comment云图制作完成")
        for _comment in reason:
            all_word += _comment["评论内容"]
        cloud_word(all_word, 'all_word')
        print("all_comment云图制作完成")
        pix = QPixmap(r'./comment_word.png')
        self.pictureLabel.setPixmap(pix)
        label = QLabel(self)
        label.setPixmap(pix)
        label.setScaledContents(True)  # 自适应QLabel大小

        layout = QVBoxLayout()
        layout.addWidget(label)
        self.setLayout(layout)
        res = bertAnalyze(analyze)
        dataAnalyzeRes = []
        for i in range(len(result)):
            if res[i] == 1:
                dataAnalyzeRes.append({"ID": str(result[i]["ID"]),
                                       "评论时间": result[i]["评论时间"],
                                       "评论内容": result[i]["评论内容"],
                                       "感情倾向": "正面"
                                       })
            elif res[i] == 0:
                dataAnalyzeRes.append({"ID": str(result[i]["ID"]),
                                       "评论时间": result[i]["评论时间"],
                                       "评论内容": result[i]["评论内容"],
                                       "感情倾向": "负面"
                                       })
        mood = pandas.DataFrame(dataAnalyzeRes)
        mood.to_excel(r'./评论情感分析.xlsx')
        print('情感分析结束')


class myThread(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("开始线程：")
        get_information()
        print("退出线程：")


# 找出字符串str中最多的字
def max_letter_count(str):
    count_max = 0
    for i in str:
        if str.count(i) > count_max:
            max_char = i
            count_max = str.count(i)
    list = []
    list.append(max_char)
    list.append(count_max)
    return list


# 爬虫
def get_information(id, headers, start_page, end_page, result, reason):
    # 目前整理的一些好评关键词
    GoodComment = {'推荐', '好用', '满意', '舒服', '喜欢', '买它', '优惠', '很值', '赞', '精美', '回购', '漂亮', '好看',
                   '不错', '新款', '实惠', '速度快', '效果好', '很舒适', '很柔软', '很合身', '真好', '继续买', '非常好',
                   '很好', '质量不错', '挺好的', '继续购买', '特别好', '蛮好', '一直在用', '非常满意', '特别好看'}
    # 输出每一页的评论
    for page in range(start_page, end_page):
        print("正在爬第" + str(page) + "页", end='')
        url = 'https://club.jd.com/comment/productPageComments.action?callback=fetchJSON_comment98&productId=' + id + '&score=0&sortType=5&page=' + str(
            page) + '&pageSize=10&isShadowSku=0&fold=1'
        data = requests.get(url, headers=headers).content.decode('gbk')
        data = data[data.find('{'):data.rfind('}') + 1]
        data = json.loads(data)
        # 筛选出有用评论
        for num in data['comments']:
            # 爬取商品评论图
            print('.', end='')
            m = 0
            path = 'photo/' + str() + str(num['id']) + '/'
            try:
                for image_url in num['images']:
                    if not os.path.exists(path):
                        os.makedirs(path)
                    if '.jpg' in image_url['imgUrl']:
                        urllib.request.urlretrieve('http:' + image_url['imgUrl'],
                                                   path + str(num['id']) + '-' + str(m) + '.jpg')
                    else:
                        urllib.request.urlretrieve('http:' + image_url['imgUrl'],
                                                   path + str(num['id']) + '-' + str(m) + '.png')
                    m = m + 1
            except:
                pass
            # 如果在某一个评论中某个字重复出现率高达30%，判断为买家的垃圾评论,不写入评论中
            if max_letter_count(num['content'])[1] / len(num['content']) > 0.3:  # 30%保证了评论字数不得低于4个
                reason.append({
                    "ID": str(num['id']),
                    "评论时间": num['creationTime'],
                    "评论内容": num['content'],
                    "理由": max_letter_count(num['content'])[0] + "重复出现率高达30%"
                })
            else:
                GoodCommentNum = 0
                for Comment in GoodComment:  # 如果评论中出现好评关键字，记录加1
                    if Comment in num['content']:
                        GoodCommentNum = GoodCommentNum + 1
                        if GoodCommentNum > 5:  # 如果好评数量大于5，判断为虚假评论
                            reason.append({
                                "ID": str(num['id']),
                                "评论时间": num['creationTime'],
                                "评论内容": num['content'],
                                "理由": "好评过多"
                            })
                if len(num['content']) < 30:  # 评论字数少于30字
                    if GoodCommentNum <= 10 and GoodCommentNum / len(
                            num['content']) <= 2 / 10:  # 如果每10字评论有好评关键字小于2,存入评论
                        result.append({
                            "ID": str(num['id']),
                            "评论时间": num['creationTime'],
                            "评论内容": num['content']
                        })
                    else:
                        reason.append({
                            "ID": str(num['id']),
                            "评论时间": num['creationTime'],
                            "评论内容": num['content'],
                            "理由": "好评过多"
                        })
                else:  # 评论字数大于30字
                    if max_letter_count(num['content'])[1] / len(num['content']) > 0.2:
                        reason.append({
                            "ID": str(num['id']),
                            "评论时间": num['creationTime'],
                            "评论内容": num['content'],
                            "理由": max_letter_count(num['content'])[0] + "重复出现率高达20%"
                        })
                    else:
                        if GoodCommentNum <= 10 and GoodCommentNum / len(
                                num['content']) < 1 / 10:  # 如果每10字评论有好评关键字小于1,存入评论
                            result.append({
                                "ID": str(num['id']),
                                "评论时间": num['creationTime'],
                                "评论内容": num['content']
                            })
                        else:
                            reason.append({
                                "ID": str(num['id']),
                                "评论时间": num['creationTime'],
                                "评论内容": num['content'],
                                "理由": "好评过多"
                            })
        print("\n爬完第" + str(page) + "页")


# 词云图生成
def cloud_word(comment_word, name):  # 生成词云
    stopwords = [line.strip() for line in open('stop_words.txt', encoding='UTF-8').readlines()]  # 加载停用词表
    comment_word = comment_word.encode('utf-8')
    comment_word = jieba.cut(comment_word)  # 未去掉停用词的分词结果
    comment_word_spite = []
    for word in comment_word:  # 去掉停分词
        if word not in stopwords:
            comment_word_spite.append(word)
    comment_word_spite = ' '.join(comment_word_spite)
    wc = wordcloud.WordCloud(  # 创建wordcloud对象
        r'msyh.ttc', width=500, height=500,
        background_color='white', font_step=2,
        random_state=False, prefer_horizontal=0.9
    )
    t = wc.generate(comment_word_spite)
    t.to_image().save('./' + name + '.png')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
