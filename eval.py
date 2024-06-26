import os
import json
import numpy as np
import pprint
import argparse
from collections import defaultdict

class EvaluationCode:
    """
    Class that would perform the following tasks to perform the
    evaluation code:
        1. Read any json file (reference or predicted) data using the json format.
        2. Compare two json files (reference ad predicted), and iterate through
        all of the generated samples and obtain the segments.
        3. Obtain the reference and predicted segments, and determine the IOU between segments (store this).
        4. From each reference segment, obtain the segments with the largest length (best predicted segments).
        5. Determine weather the emotion is matching with the corresponding predicted segment.
        6. Select the best reference and user segment based on the highest length and store this.
        7. Create a json file that would output, per corresponding pair, the IOUs, and also the emotion precision
    """

    def __init__(self, ref_path:str, pred_path:str, save_path: str, num_segments=None):
        """
        Cosntructor that would set the path for the reference
        and predicted jsons.
        """
        #Set the paths
        self.ref_json = self._json_file_read(ref_path)
        self.pred_json = self._json_file_read(pred_path)
        self.save_path = save_path
        self.num_segments = num_segments

    def execute(self):
        """
        Instace method that would perform all of
        the necessary processes to obtain the results.
        """

        #Create placeholders for ref data
        ref_num_segments = []

        #Create placeholders for pred data
        pred_num_segments = []

        #Iterate through each of the ref and pred files
        num_gt_segments = 0
        num_matched_segments = 0
        iou_list_total = []
        iou_list_matched = []
        num_correct_emotions_in_matched = 0
        num_pred_segments = 0
        iou_emotions_total = []

        for file in self.ref_json:
            
            #Obtain the metadata of ref and pred json files
            ref_segment_metadata = self.ref_json[file]
            pred_segment_metadata = self.pred_json[file]

            if self.num_segments is not None and ref_segment_metadata["num_segments"] != self.num_segments:
                continue

            #Determine the number of segments
            ref_num_segments.append(ref_segment_metadata["num_segments"])
            pred_num_segments.append(pred_segment_metadata["num_segments"])
            num_pred_segments += pred_segment_metadata["num_segments"]
            assert pred_segment_metadata["num_segments"] == len(pred_segment_metadata["segments"])
            assert ref_segment_metadata["num_segments"] == len(ref_segment_metadata["segments"])


            #Obtain the user and reference segments
            ref_segments = ref_segment_metadata["segments"]
            pred_segments = pred_segment_metadata["segments"]

            # remove spaces and make all characters lower case
            ref_segments = [{"Emotion": x["Emotion"], "Segment": x["Segment"].replace(" ", "").lower()} for x in ref_segments]
            pred_segments = [{"Emotion": x["Emotion"], "Segment": x["Segment"].replace(" ", "").lower()} for x in pred_segments]

            # for debubbing purposes
            # ref_sentence = ""
            # pred_sentence = ""
            # for seg in ref_segments:
            #     ref_sentence += seg["Segment"]
            # for seg in pred_segments:
            #     pred_sentence += seg["Segment"]
            # if ref_sentence != pred_sentence:
            #     breakpoint()

            #Obtain all emotions from ref_segments
            emotions_ref_segments =  [x["Emotion"].upper() for x in ref_segments]
            emotions_pred_segments = [x["Emotion"].upper() for x in pred_segments]

            #Calculate the Iou between emotions_ref_segments and emotions_pred_segments
            iou_emotions_total.append(self._calculate_iou(np.array(emotions_ref_segments), np.array(emotions_pred_segments)))
            
            #Iterate through ref and pred segments
            ref_idx_start = 0
            used_preds = set()
            for ref_seg in ref_segments:

                num_gt_segments += 1
                
                #Iterate now between the pred segments
                tmp_ious_compare = []
                ref_idx_end = ref_idx_start + len(ref_seg["Segment"])
                pred_idx_start = 0

                seg_num = 0
                for pred_seg in pred_segments:

                    pred_idx_end = pred_idx_start + len(pred_seg["Segment"])

                    # skipping already matched segments
                    if seg_num in used_preds:
                        pred_idx_start = pred_idx_end
                        tmp_ious_compare.append(0)
                        seg_num += 1
                        continue

                    # compute overlap between them
                    range1 = np.arange(pred_idx_start, pred_idx_end)
                    range2 = np.arange(ref_idx_start, ref_idx_end)

                    #Calculate the Iou between pred_segment and ref seg
                    tmp_ious_compare.append(self._calculate_iou(range1, range2))

                    pred_idx_start = pred_idx_end
                    seg_num += 1

                # update start of next reference
                ref_idx_start = ref_idx_end

                #Cast to np array
                tmp_ious_compare = np.array(tmp_ious_compare)

                #Find the index where maximum is
                max_iou_index = np.argmax(tmp_ious_compare)

                if len(tmp_ious_compare) == 0 or tmp_ious_compare[max_iou_index] < 0.5:
                    no_match = True
                    iou_list_total.append(tmp_ious_compare[max_iou_index])
                    continue
                else:
                    no_match = False
                    used_preds.add(max_iou_index)
                    num_matched_segments += 1
                    iou_list_total.append(tmp_ious_compare[max_iou_index])
                    iou_list_matched.append(tmp_ious_compare[max_iou_index])

                if ref_seg["Emotion"].upper() == pred_segments[max_iou_index]["Emotion"]:
                    num_correct_emotions_in_matched += 1

        #Create two distionaries of data, reference and matched
        eval_metrics_total = {
            "num_gt_segments": num_gt_segments,
            "IoU_total": np.mean(np.array(iou_list_total)),
            "IoU_matched": np.mean(np.array(iou_list_matched)),
            "IoU_total_emotions": np.mean(np.array(iou_emotions_total)),
            "emotion_accuracy_matched": num_correct_emotions_in_matched / num_gt_segments,
            "segmentation_recall": num_matched_segments / num_gt_segments,
            "segmentation_precision": num_matched_segments / num_pred_segments
        }

        os.makedirs(os.path.dirname(self.save_path), exist_ok = True)
        with open(self.save_path, 'w') as f:
            json.dump(eval_metrics_total, f, indent=6)

        return eval_metrics_total
            
    def _split_string(self, s: str):
        """
        Instance method that would split a string
        if it contains a space, otherwise, return the
        single string element in a list
        """
        if ' ' in s:
            return np.array(s.split())
        else:
            return np.array([s])

    def _json_file_read(self, path:str):
        """
        Instance method that would read the files
        based on the passed paths
        """
        # Read file with conext manager
        with open(path, "r") as read_file:
            json_file = json.load(read_file)
        return json_file

    def _calculate_iou(self, vector1:np.ndarray, vector2:np.ndarray):
        """
        class method method to obtain the intersection 
        over union for two np.array vectors that contains strings
        that are already split
        """        
        # Calculate intersection
        intersection = np.intersect1d(vector1, vector2)
        
        # Calculate union
        union = np.union1d(vector1, vector2)
        
        # Calculate IoU
        iou = len(intersection) / len(union)
        
        return iou

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=str, required=True)
    parser.add_argument("--gt-path", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--num-segments", type=int, default=None)
    args = parser.parse_args()

    #Create the data path to the files
    # predicted_path = os.path.join(os.path.dirname(__file__), "lstm_lr=1e-3_adam.json")
    # ground_truth_path = os.path.join(os.path.dirname(__file__), "test_gt.json")

    #Create eval compiler
    eval_compiler = EvaluationCode(pred_path=args.pred_path, ref_path=args.gt_path, \
        save_path=args.save_path, num_segments=args.num_segments)
    eval_metrics = eval_compiler.execute()

    pprint.pprint(eval_metrics)