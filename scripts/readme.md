====================== 换脸的大致流程 ===========================
假设有图片source和target，为了实现影视级别的换脸（即将source的脸完整地换到target上去），大致需要以下几步：

（1）将source的 mask， 按照target的样子进行驱动，这一步我们用了 face_vid2vid 来完成（face_vid2vid在眼神处理的不是很好，但整体上驱动的还算ok）；
（2）将 face_vid2vid 驱动的结果用超分辨率增强，因为 face_vid2vid 的结果是256的，而且存在一些artifacts，这一步我们用 GPEN 来增强；
（3）将驱动并增强后的图片，我们只提取对应的mask，随后贴到target上去，相当于换mask；
（4）最后用原始srouce 的 各个部位的style + 替换后的mask，生成图片，完成换脸。
================================================================


====================== 换脸的具体流程 ===========================
****** 步骤（1） ******
具体见 /apdcephfs/share_1290939/zhianliu/py_projects/our_editing/swap_face_fine/face_vid2vid/drive_demo.py 中的 drive_source_demo()

对应的测试用例见 /apdcephfs/share_1290939/zhianliu/py_projects/our_editing/dummy.py 中的 test_facevid2vid_demo() 

****** 步骤（2） ******
具体见 /apdcephfs/share_1290939/zhianliu/py_projects/our_editing/swap_face_fine/gpen/gpen_demo.py 中的 GPEN_demo()

对应的测试用例见 /apdcephfs/share_1290939/zhianliu/py_projects/our_editing/dummy.py 中的 test_GPEN_demo() 

****** 步骤（3） ******
（a）将步骤（2）中得到的图片，用face parser提取mask

具体见 /apdcephfs/share_1290939/zhianliu/py_projects/our_editing/swap_face_fine/face_parsing/face_parsing_demo.py 中的 faceParsing_demo(), 注意此时 送到网络的输入是 1024 尺寸的

（b）现在得到了驱动后图片的mask ，准备与target图片原始的mask进行交换头部区域（注意，需要保留target的hair、belowface、background，而使用驱动后图片脸部的mask），参见 swap_face_mask.py 

换了mask之后皮肤 和 hair处会有一些洞，我们填充皮肤.

****** 步骤（4） ******
用步骤（2）中得到的driven image脸部的style code来 + 步骤（3.b）之后的脸部mask 来编辑脸部，同时，使用保持target头发区域的style， 具体见 ./Face_swap_with_two_imgs.py
