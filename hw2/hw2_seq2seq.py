import tensorflow as tf
import numpy as np
import os
import sys
import time
from six.moves import cPickle
from model import Video_Caption_Generator
import json

def main():
   
    testing_file = sys.argv[1] + 'testing_id.txt'#file which contains all testing video ids
    testing_path = sys.argv[1] + 'testing_data/feat/'#directory which contains testing video feature files
    result_file = sys.argv[2]#result txt file
    
    peer_review_file = sys.argv[1] + 'peer_review_id.txt'
    peer_review_path = sys.argv[1] + 'peer_review/feat/'
    peer_review_result_file = sys.argv[3]

#    testing_file = 'MLDS_hw2_data/testing_id.txt'#file which contains all testing video ids
#    testing_path = 'MLDS_hw2_data/testing_data/feat/'#directory which contains testing video feature files
#    result_file = 'test1001'#result txt file
#    
##    peer_review_file = sys.argv[1] + 'peer_review_id.txt'
##    peer_review_path = sys.argv[1] + 'peer_review/feat/'
##    peer_review_result_file = sys.argv[3]
    
    init_from = 'model_final'#model save
   
    prediction(testing_file, testing_path, result_file, init_from)
    prediction(peer_review_file, peer_review_path, peer_review_result_file, init_from)
    
    
          
def get_testing_feat(feat_path, batch_video_id):
	feature = []
	feature_mask = []
	for i in range(len(batch_video_id)):
		fn = os.path.join(feat_path, batch_video_id[i]+'.npy')
		if os.path.exists(fn):
			d = np.load(fn)
			feature.append(d)
			feature_mask.append(np.ones(d.shape[0]))
		else:
			print('%s does not exist' % (fn))
	feature = np.asarray(feature)
	feature_mask = np.asarray(feature_mask)
	return feature, feature_mask


def prediction(testing_file, testing_path, result_file, init_from):

    assert os.path.isfile(os.path.join(init_from,"config.pkl")), "config.pkl file does not exist in path %s" % init_from
	# open old config and check if models are compatible
    with open(os.path.join(init_from, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)

    with open(os.path.join(init_from, 'vocab.pkl'), 'rb') as f:
        vocab = cPickle.load(f)

    vocab_inv = {v:k for k, v in vocab.items()}

    with open(testing_file,'r') as f:
        test_feat_id = f.readlines()
        for i in range(len(test_feat_id)):	       
            test_feat_id[i] = test_feat_id[i].replace('\n','')

    model = Video_Caption_Generator(saved_args,n_vocab=len(vocab),infer=True)
	
    with tf.Session() as sess:
        result = []
        for i in range(len(test_feat_id)):
#        for i in range(0,1):
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(init_from)

            if ckpt and ckpt.model_checkpoint_path: # args.init_from is not None:
                saver.restore(sess, ckpt.model_checkpoint_path)
                if i == 0:
                    print("Model restored %s" % ckpt.model_checkpoint_path)
            sess.run(tf.global_variables())

			
            this_test_feat_id = test_feat_id[i]

			# get vdieo features
			# notes: the second argument to get_video_feat must be np.array
            current_feat, current_feat_mask = get_testing_feat(testing_path, np.array([this_test_feat_id]))
			
            this_gen_idx, probs = sess.run([model.gen_caption_idx,model.pred_probs],feed_dict={
										model.video: current_feat,
										model.video_mask : current_feat_mask
										})

            this_gen_words = []

            for k in range(len(this_gen_idx)):
                this_gen_words.append(vocab_inv.get(this_gen_idx[k],'<PAD>'))


            this_gen_words = np.array(this_gen_words)

            punctuation = np.argmax(this_gen_words == '<EOS>') + 1
			
            if punctuation > 1:
                this_gen_words = this_gen_words[:punctuation]


            this_caption = ' '.join(this_gen_words)
            this_caption = this_caption.replace('<BOS> ', '')
            this_caption = this_caption.replace(' <EOS>', '')

            this_answer = {}
            this_answer['caption'] = this_caption
            this_answer['id'] = this_test_feat_id

            print('Id: %s, caption: %s' % (this_test_feat_id, this_caption))

            result.append(this_answer)
        
        output_caption = []
        output_feat_id = []
        for i in range(len(result)):
            output_caption.append(result[i]['caption'])
            output_feat_id.append(result[i]['id'])
        
        
        output_captionv1 = []
        for j in range(0,len(output_caption)):
            zzz = output_caption[j]
        #    for i in range(0,len(zzz)):
            temp_z1 = []
            temp_z = zzz.split(" ")
#        if (temp_z[1]=='is' or temp_z[1]=='and' or temp_z[1]=='are'):
#            temp_z.insert(1,'man')
            temp_z.insert(1,'man')
            for i in range(0,len(temp_z)):
#            if(temp_z[0]=='a'):
#                temp_z[0]=='A'
                if (temp_z[i]!='<PAD>'and i <= 7):
                    temp_z1.append(temp_z[i])
                    temp_z1[0] = 'A'
                    str1 = ' '.join(temp_z1)
                
            output_captionv1.append(str1)
        
        
        
        special_resu = []            
        for i_result in range(0,len(output_feat_id)):
            temp_resu = [output_feat_id[i_result],output_captionv1[i_result]]
            special_resu.append(temp_resu)

        with open(result_file, "w") as text_file:
            for i_resu in range(0,len(special_resu)):
                text_file.write(str(special_resu[i_resu][0])+","+str(special_resu[i_resu][1])) 
                text_file.write('\n')
        

if __name__ == '__main__':
    main()
