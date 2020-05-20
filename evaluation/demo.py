import evaluation 
import cv2

if __name__ == '__main__':
    test_mask = 'test_mask.png'

    mask_np = cv2.imread(test_mask)/255.0
    mask_np = cv2.resize(mask_np, (256,256))
    cv2.imwrite(test_mask, mask_np)

    test_ibl  = 'test_ibl.png'
    out_file = 'test_pred.png'
    evaluation.net_render(test_mask, test_ibl, out_file, save_npy=True)
