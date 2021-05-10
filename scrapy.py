#!/usr/bin/python
# -*- coding: utf-8 -*-
import re
import time
import xlsxwriter
from selenium import webdriver
from bs4 import BeautifulSoup
import io
import sys


def qq_login(gid):
    """
    通过selenium模拟登录，获取群成员信息
    :param gid: 群号
    :return: 页面源码
    """
    driver = webdriver.Chrome()
    driver.get('https://qun.qq.com/member.html#gid=%s' % gid)
    driver.maximize_window()
    time.sleep(3)
    # 切换iframe授权登录
    driver.switch_to_frame('login_frame')
    driver.find_element_by_class_name('img_out_focus').click()

    # 拉动滑动条加载剩余数据
    for i in range(1000):
        js = "window.scrollTo(0,document.body.scrollHeight)"
        driver.execute_script(js)

    # 获取页面源码并写入缓存
    res = driver.page_source
    driver.close()
    return res


def dispose(res):
    """
    处理页面源码数据，提取群成员信息
    :param res: 页面源码
    :return: 处理后的list_a列表
    """
    # 改变标准输出的默认编码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='gb18030')

    soup = BeautifulSoup(res, 'lxml')

    # 查找tr标签下的class属性包含‘mb’的节点树
    c = soup.find_all('tr', attrs={"class": re.compile('mb')})
    list_a = []
    for i in c:
        # 处理一些特殊字符‘\n’,'\t',替换成‘,’
        str_a = i.text.replace('\n', '').replace('\t', ',')
        # 通过正则表达式，切割以‘,’分割的字符串组成列表
        data = re.split(r',', str_a)
        # 去除空字符""
        data_list = [i for i in data if i != '']
        # 获取头像地址并插入列表
        img = "https:" + i.img.get('src')

        # 将头像地址插入列表中第3个位置的元素
        data_list.insert(2, img)
        # 删除列表中的序号
        del data_list[0]
        # 由于前面删除了空字符"",导致有些人的群昵称为空也删除，这里手动添加进去。
        if len(data_list) < 9:
            data_list.insert(2, '')

        # 每一个成员信息为一个列表，添加到list_a作为元素
        list_a.append(data_list)
    return list_a


def write_execl(list_a):
    """
    写入execl表格
    :param list_a: 数据列表
    :return:
    """
    if len(list_a) > 2:

        # 创建execl
        new_time = time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())
        workbook = xlsxwriter.Workbook('{}.xlsx'.format(new_time))  # 新建excel表
        worksheet = workbook.add_worksheet('sheet1')  # 新建sheet（sheet的名称为"sheet1"）

        bold = workbook.add_format({
            'bold': 1,  # 字体加粗
            'fg_color': 'green',  # 单元格背景颜色
            'align': 'center',  # 对齐方式
            'valign': 'vcenter',  # 字体对齐方式
        })

        # 写表头
        work_header = ['QQ昵称', '头像地址', '群昵称', 'QQ号', '性别', 'Q龄', '入群时间', '等级（积分）', '最后发言']
        worksheet.write_row('A1', work_header, bold)

        # 遍历多少条数据就写入多少行数据到execl，表头已经占了A1，所以从A2开始写入，index从0开始遍历（2+index）
        for index in range(len(list_a)):
            worksheet.write_row('A%s' % (2 + index), list_a[index])

        # 最后关闭workbook,否则不会产生execl文件
        workbook.close()
    else:
        print('请检查群号是否有误，没有获取到群成员信息，放弃写入execl')


if __name__ == '__main__':
    res = qq_login('填写群号')
    list_a = dispose(res)
    write_execl(list_a)
