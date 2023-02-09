[1mdiff --git a/bspytasks/spoken_digit_task/ti_spoken_digits.py b/bspytasks/spoken_digit_task/ti_spoken_digits.py[m
[1mindex 5c0f9db..dd803e5 100644[m
[1m--- a/bspytasks/spoken_digit_task/ti_spoken_digits.py[m
[1m+++ b/bspytasks/spoken_digit_task/ti_spoken_digits.py[m
[36m@@ -69,7 +69,6 @@[m [mclass FCLayer(torch.nn.Module):[m
         out = self.dropout(out)  [m
 [m
         out = self.fc2(out)[m
[31m-        [m
         out = torch.log_softmax(out, dim=1)[m
         return out[m
 [m
[36m@@ -199,8 +198,11 @@[m [mclass ProjectedAudioDataset(Dataset):[m
                 downsample_method[m
                 ) -> None:[m
 [m
[31m-        self.mean = 0.[m
[31m-        self.std = 0.[m
[32m+[m[32m        # EXTRACTED FROM TRAINING DATASET ONLY[m
[32m+[m[32m        self.mean = 0.01284[m
[32m+[m[32m        self.std = 1.02609[m
[32m+[m
[32m+[m[32m        # self.mean, self.std = 0., 0.[m
 [m
         self.transform = transform[m
         self.num_downsample = num_downsample[m
[36m@@ -227,12 +229,17 @@[m [mclass ProjectedAudioDataset(Dataset):[m
                         avg = np.mean([m
                             tmp[i][slope_length + rest_length - 100 : slope_length + rest_length, 0][m
                         )[m
[32m+[m[32m                        avg2 = np.mean([m
[32m+[m[32m                            data[m
[32m+[m[32m                        )[m
                         if top_projections != None:[m
                             if i in top_projections:[m
[31m-                                self.dataset_list.append(np.append((data - avg), i))[m
[32m+[m[32m                                self.dataset_list.append(np.append((data - avg2), i))[m
[32m+[m[32m                                # self.dataset_list.append(np.append(data, i))[m
                         else:[m
                             if len(data) < 10000 and len(data) > 2000:[m
[31m-                                self.dataset_list.append(np.append((data - avg), i))[m
[32m+[m[32m                                self.dataset_list.append(np.append((data - avg2), i))[m
[32m+[m[32m                                # self.dataset_list.append(np.append(data, i))[m
                         if len(data) > self.max_length:[m
                             self.max_length = len(data)[m
                         if len(data) < self.min_legnth:[m
[36m@@ -260,6 +267,34 @@[m [mclass ProjectedAudioDataset(Dataset):[m
             self.len_dataset[m
         ))[m
 [m
[32m+[m[32m        # Calculating mean and std[m
[32m+[m[32m        # self.mean = np.average(self.dataset_numpy[:, :-1])[m
[32m+[m[32m        # self.std  = np.std(self.dataset_numpy[:, :-1])[m
[32m+[m[32m        # # Normalizing dataset[m
[32m+[m[32m        # self.dataset_numpy[:, :-1] = ((self.dataset_numpy[:,:-1]) - self.mean)/self.std[m
[32m+[m
[32m+[m[32m        # means = [][m
[32m+[m[32m        # stds = [][m
[32m+[m
[32m+[m[32m        # for i in range(len(self.dataset_list)):[m
[32m+[m[32m        #     means.append([m
[32m+[m[32m        #         np.mean(self.dataset_list[i])[m
[32m+[m[32m        #     )[m
[32m+[m[32m        #     stds.append([m
[32m+[m[32m        #         np.std(self.dataset_list[i])[m
[32m+[m[32m        #     )[m
[32m+[m
[32m+[m[32m        # self.mean = np.mean(means)[m
[32m+[m[32m        # # self.std = np.mean(stds)[m
[32m+[m[32m        # tmp = 0.[m
[32m+[m[32m        # for i in range(len(stds)):[m
[32m+[m[32m        #     tmp += stds[i]**2[m
[32m+[m[32m        # tmp /= len(stds)[m
[32m+[m[32m        # self.std = tmp ** 0.5[m
[32m+[m
[32m+[m[32m        # for i in range(len(self.dataset_list)):[m
[32m+[m[32m        #     self.dataset_list[i] = (self.dataset_list[i] - self.mean)/self.std[m
[32m+[m
         if downsample_method == 'variable':[m
             for i in range(0, self.len_dataset):[m
                 projection_idx = self.dataset_list[i][-1][m
[36m@@ -302,13 +337,6 @@[m [mclass ProjectedAudioDataset(Dataset):[m
         else:[m
             print("Downsample method UNKNOWN!")[m
         [m
[31m-[m
[31m-        # Calculating mean and std[m
[31m-        self.mean = np.average(self.dataset_numpy[:, :-1])[m
[31m-        self.std  = np.std(self.dataset_numpy[:, :-1])[m
[31m-[m
[31m-        self.dataset_numpy[:, :-1] = ((self.dataset_numpy[:,:-1]) - self.mean)/self.std[m
[31m-[m
         print("Loading completed successfully!")[m
         print("Lenght of dataset: ", self.len_dataset)[m
         print("---------------------------------------------------")[m
[36m@@ -352,6 +380,115 @@[m [mdef reset_weights(m):[m
             print("Reset trainable parameters of layer = ", layer)[m
             layer.reset_parameters()[m
 [m
[32m+[m[32mdef train_and_test([m
[32m+[m[32m        model,[m
[32m+[m[32m        num_epoch,[m
[32m+[m[32m        dataset,[m
[32m+[m[32m        batch_size,[m
[32m+[m[32m):[m
[32m+[m[32m    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')[m
[32m+[m[32m    model.to(torch.float)[m
[32m+[m[32m    model.to(device)[m
[32m+[m[32m    model = torch.compile(model)[m
[32m+[m[32m    scaler = torch.cuda.amp.GradScaler(enabled=True)[m
[32m+[m[32m    loss_fn = torch.nn.CrossEntropyLoss()[m
[32m+[m
[32m+[m[32m    train_set, test_set = torch.utils.data.random_split([m
[32m+[m[32m                            dataset,[m
[32m+[m[32m                            [[m
[32m+[m[32m                                int(0.8 * len(dataset)),[m[41m [m
[32m+[m[32m                                int(0.2 * len(dataset) + 1)[m
[32m+[m[32m                            ][m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    train_dataloader = DataLoader([m
[32m+[m[32m        train_set,[m
[32m+[m[32m        batch_size,[m
[32m+[m[32m        drop_last= True,[m
[32m+[m[32m        shuffle= True,[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    test_dataloader = DataLoader([m
[32m+[m[32m        test_set,[m
[32m+[m[32m        batch_size,[m
[32m+[m[32m        drop_last= True,[m
[32m+[m[32m        shuffle= True,[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay = .1)[m
[32m+[m[32m    scheduler = torch.optim.lr_scheduler.OneCycleLR([m
[32m+[m[32m                                                optimizer,[m[41m [m
[32m+[m[32m                                                max_lr = 0.01,[m
[32m+[m[32m                                                steps_per_epoch = int(len(train_dataloader)),[m
[32m+[m[32m                                                epochs = num_epoch,[m
[32m+[m[32m                                                anneal_strategy = 'linear'[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    model.train()[m
[32m+[m[32m    for epoch in range(0, num_epoch):[m
[32m+[m[32m        print("Starting epoch: ", epoch + 1)[m
[32m+[m[32m        current_loss = 0.[m
[32m+[m[32m        for i, data in enumerate(train_dataloader):[m
[32m+[m[32m            inputs = data['audio_data'].to(device)[m
[32m+[m[32m            targets = data['audio_label'].type(torch.LongTensor).to(device)[m
[32m+[m[32m            optimizer.zero_grad()[m
[32m+[m
[32m+[m[32m            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):[m
[32m+[m[32m                outputs = model(inputs)[m
[32m+[m[32m                loss = loss_fn(outputs, targets)[m
[32m+[m
[32m+[m[32m            scaler.scale(loss).backward()[m
[32m+[m[32m            scaler.step(optimizer)[m
[32m+[m[32m            scheduler.step()[m
[32m+[m[32m            scaler.update()[m
[32m+[m
[32m+[m[32m            current_loss += loss.item()[m
[32m+[m[32m            if i % (batch_size) == (batch_size - 1):[m
[32m+[m[32m                print("Loss after mini-batch %3d: %.3f" % (i+1, current_loss/(5 * batch_size)))[m
[32m+[m[32m                current_loss = 0.[m
[32m+[m[41m    [m
[32m+[m[32m    print("Evaluating training procedure...")[m
[32m+[m[32m    model.eval()[m
[32m+[m[32m    correct, total = 0., 0.[m
[32m+[m[32m    with torch.no_grad():[m
[32m+[m[32m        for i, data in enumerate(test_dataloader):[m
[32m+[m[32m            inputs = data['audio_data'].to(device)[m
[32m+[m[32m            targets = data['audio_label'].type(torch.LongTensor).to(device)[m
[32m+[m[32m            outputs = model(inputs)[m
[32m+[m[32m            _, predicted = torch.max(outputs, 1)[m
[32m+[m[32m            total += targets.size(0)[m
[32m+[m[32m            correct += (predicted==targets).sum().item()[m
[32m+[m[32m        print("Training accuracy: ", 100.*correct/total)[m
[32m+[m
[32m+[m[32m    torch.save(model.state_dict(), "saved_model.pt")[m
[32m+[m[32m    np.save("test_set.npy", test_set)[m
[32m+[m
[32m+[m[32m    print(" ")[m
[32m+[m
[32m+[m[32mdef test([m
[32m+[m[32m    model,[m
[32m+[m[32m    dataset,[m
[32m+[m[32m    device[m
[32m+[m[32m    ):[m
[32m+[m
[32m+[m[32m    test_dataloader = DataLoader([m
[32m+[m[32m        dataset,[m
[32m+[m[32m        batch_size= 1,[m
[32m+[m[32m        shuffle= False,[m
[32m+[m[32m        drop_last= False[m
[32m+[m[32m    )[m
[32m+[m[32m    print("Length of dataset: ", len(test_dataloader))[m
[32m+[m[32m    correct, total = 0, 0[m
[32m+[m[32m    with torch.no_grad():[m
[32m+[m[32m        for i, data in enumerate(test_dataloader):[m
[32m+[m[32m            inputs = data['audio_data'].to(device)[m
[32m+[m[32m            targets = data['audio_label'].to(device)[m
[32m+[m[32m            outputs = model(inputs)[m
[32m+[m[32m            _, predicted = torch.max(outputs, 1)[m
[32m+[m[32m            total += targets.size(0)[m
[32m+[m[32m            correct += (predicted == targets).sum().item()[m
[32m+[m[32m        # print("Number of tested data: ", i)[m
[32m+[m[32m        print("Test accuracy: ", 100. * correct / total)[m
 [m
 def kfold_cross_validation([m
         model,[m
[36m@@ -396,8 +533,7 @@[m [mdef kfold_cross_validation([m
         )[m
 [m
         model.apply(reset_weights)[m
[31m-[m
[31m-        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay = 1)[m
[32m+[m[32m        optimizer = torch.optim.AdamW(model.parameters(), lr = 0.01, weight_decay = .1)[m
         scheduler = torch.optim.lr_scheduler.OneCycleLR([m
                                                     optimizer, [m
                                                     max_lr = 0.01,[m
[36m@@ -407,10 +543,9 @@[m [mdef kfold_cross_validation([m
         )[m
 [m
         for epoch in range(0, num_epoch):[m
[31m-            if epoch % 10 == 0:[m
[31m-                print("Starting epoch ", epoch+1)[m
[32m+[m[32m            print("Starting epoch ", epoch+1)[m
             current_loss = 0.[m
[31m-            for i, data in enumerate(trainloader):[m
[32m+[m[32m            for i, data in enumerate(trainloader, 0):[m
                 inputs = data['audio_data'].to(device)[m
                 targets = data['audio_label'].type(torch.LongTensor).to(device)[m
                 optimizer.zero_grad()[m
[36m@@ -429,20 +564,12 @@[m [mdef kfold_cross_validation([m
                     print("Loss after mini-batch %3d: %.3f" % (i+1, current_loss/(5 * batch_size)))[m
                     current_loss = 0.[m
         [m
[31m-        print("Training process completed, saving model ...")[m
[31m-[m
[31m-        #TODO: Save model[m
[31m-[m
[31m-        print("Start testing...")[m
[31m-[m
[31m-        # Evaluation for this fold[m
[32m+[m[32m        print("Start testing for fold: ", fold)[m
         correct, total = 0, 0[m
         with torch.no_grad():[m
[31m-            for i, data in enumerate(testloader):[m
[31m-                # inputs, targets = data[0].to(device), data[1].to(device)[m
[32m+[m[32m            for i, data in enumerate(testloader, 0):[m
                 inputs = data['audio_data'].to(device)[m
                 targets = data['audio_label'].type(torch.LongTensor).to(device)[m
[31m-[m
                 outputs = model(inputs)[m
                 _, predicted = torch.max(outputs, 1)[m
                 total += targets.size(0)[m
[36m@@ -450,58 +577,6 @@[m [mdef kfold_cross_validation([m
             print("Accuracy for fold %d: %d %%" %(fold, 100.*correct/total))[m
             print('------------------------------------')[m
             results[fold] = 100. * (correct / total)[m
[31m-[m
[31m-        # A list to keep track of projection scores; 1 -> correct, 0 -> incorrect[m
[31m-        # [projection_idx, 0/1, target_digit][m
[31m-        # projection_score_list = [][m
[31m-        # with torch.no_grad():[m
[31m-        #     for i, data in enumerate(testloader):[m
[31m-        #         inputs, targets = data[0].to(device), data[1].to(device)[m
[31m-        #         outputs = model(inputs[:,:-1])[m
[31m-        #         _, predicted = torch.max(outputs, 1)[m
[31m-[m
[31m-        #         for i in range(predicted.size(0)):[m
[31m-        #             if predicted[i] == targets[i]:[m
[31m-        #                 projection_score_list.append([m
[31m-        #                     [inputs[i, -1], 1, targets[i]][m
[31m-        #                 )[m
[31m-        #             else:[m
[31m-        #                 projection_score_list.append([m
[31m-        #                     [inputs[i, -1], 0, targets[i]][m
[31m-        #                 )[m
[31m-[m
[31m-            # Sorting projection_score_list based on target digit[m
[31m-            # [[projection_idx, corr./incorr., target_digit]][m
[31m-            # accuracy_list_sort_by_target = [][m
[31m-            # for i in range(0, num_classes):[m
[31m-            #     temp = [][m
[31m-            #     for j in range(len(projection_score_list)):[m
[31m-            #         if projection_score_list[j][2] == i:[m
[31m-            #             temp.append([projection_score_list[j][0], projection_score_list[j][1], i])[m
[31m-            #     accuracy_list_sort_by_target.append(temp)[m
[31m-            [m
[31m-            # Voting mechanism[m
[31m-            # [Correct predictions, total predictions][m
[31m-            # voting_accuracy_of_digits = [][m
[31m-            # for i in range(0, num_classes):[m
[31m-            #     temp = 0[m
[31m-            #     for j in range(len(accuracy_list_sort_by_target[i])):[m
[31m-            #         temp += accuracy_list_sort_by_target[i][j][1][m
[31m-            #     voting_accuracy_of_digits.append([temp, len(accuracy_list_sort_by_target[i])])[m
[31m-                # if temp >= len(accuracy_list_sort_by_target[i])//2:[m
[31m-                    # voting_accuracy_of_digits.append(1)[m
[31m-                # else:[m
[31m-                    # voting_accuracy_of_digits.append(0)[m
[31m-[m
[31m-            # Sorting proejction score list based on projections[m
[31m-            # Here we can find best projections[m
[31m-            # accuracy_list_sort_by_projection_idx = [][m
[31m-            # for i in range(0, num_projections):[m
[31m-            #     temp = [][m
[31m-            #     for j in range(len(projection_score_list)):[m
[31m-            #         if projection_score_list[j][0] == i:[m
[31m-            #             temp.append([projection_score_list[j][0], projection_score_list[j][1], projection_score_list[j][2]])[m
[31m-            #     accuracy_list_sort_by_projection_idx.append(temp)[m
         [m
     print(f"K-FOLD cross validation results for {k_folds} FOLDS")[m
     print('------------------------------------')[m
[36m@@ -511,7 +586,9 @@[m [mdef kfold_cross_validation([m
         sum += value[m
     print(f"Average accuracy: {sum / len(results.items())} %")[m
     print("Saving model ... ")[m
[32m+[m
     torch.save(model.state_dict(), "saved_model.pt")[m
[32m+[m[32m    np.save("test_set.npy", test_sampler)[m
 [m
     print(" ")[m
 [m
[36m@@ -644,13 +721,14 @@[m [mif __name__ == '__main__':[m
 [m
     batch_size = 128[m
 [m
[31m-    num_epoch = 25[m
[32m+[m[32m    num_epoch = 50[m
 [m
     num_classes = 10[m
     normalizing_dataset = True[m
     train_with_all_projections = True[m
     new_number_of_projetions = 64[m
[31m-    zero_padding_downsample = True    [m
[32m+[m[32m    zero_padding_downsample = True[m[41m  [m
[32m+[m
     # np.random.seed(5) [m
     # configs = load_configs('configs/defaults/processors/hw.yaml')[m
     # measurement([m
[36m@@ -681,18 +759,22 @@[m [mif __name__ == '__main__':[m
 [m
     empty = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_ti_alpha/boron_roomTemp_30nm/ti_spoken_digits/test/empty/"[m
 [m
[31m-[m
     dataset_path = ([m
[32m+[m[32m        empty,[m
         projected_train_val_data_arsenic_f4,[m
[31m-        # projected_train_val_data_arsenic_f5,[m
[31m-        # projected_train_val_data_arsenic_f6,[m
[31m-        # projected_train_val_data_arsenic_f7,[m
[31m-        # projected_train_val_data_arsenic_f8,[m
[32m+[m[32m        projected_train_val_data_arsenic_f5,[m
[32m+[m[32m        projected_train_val_data_arsenic_f6,[m
[32m+[m[32m        projected_train_val_data_arsenic_f7,[m
[32m+[m[32m        projected_train_val_data_arsenic_f8,[m
[32m+[m[32m    )[m
[32m+[m
[32m+[m[32m    test_dataset_path = ([m
         projected_test_data_arsenic_f4,[m
[31m-        # projected_test_data_arsenic_f5,[m
[31m-        # projected_test_data_arsenic_f6,[m
[31m-        # projected_test_data_arsenic_f7,[m
[31m-        # projected_test_data_arsenic_f8[m
[32m+[m[32m        projected_test_data_arsenic_f5,[m
[32m+[m[32m        projected_test_data_arsenic_f6,[m
[32m+[m[32m        projected_test_data_arsenic_f7,[m
[32m+[m[32m        projected_test_data_arsenic_f8,[m
[32m+[m
     )[m
 [m
     transform = transforms.Compose([[m
[36m@@ -707,28 +789,65 @@[m [mif __name__ == '__main__':[m
                 slope_length        = slope_length,[m
                 rest_length         = rest_length,[m
                 num_downsample      = down_sample_no,[m
[31m-                downsample_method  = 'zero_pad_sym' # 'variable', 'zero_pad', 'zero_pad_sym'[m
[32m+[m[32m                downsample_method  = 'zero_pad' # 'variable', 'zero_pad', 'zero_pad_sym'[m
                 # NOTE: Variable length zero padding is logically incorrect,[m
[31m-                # the reason is that it is basically means varible low-pass filtering, high for low-durated audios, and low for high-duration adious[m
[32m+[m[32m                # Because it means varible low-pass filtering, high for low-durated audios, and low for high-duration adious[m
     )[m
[31m-    [m
 [m
[32m+[m[32m    # test_dataset = ProjectedAudioDataset([m
[32m+[m[32m    #             data_dirs           = test_dataset_path,[m
[32m+[m[32m    #             transform           = transform,[m
[32m+[m[32m    #             num_projections     = 128,[m
[32m+[m[32m    #             top_projections     = None,[m
[32m+[m[32m    #             slope_length        = slope_length,[m
[32m+[m[32m    #             rest_length         = rest_length,[m
[32m+[m[32m    #             num_downsample      = down_sample_no,[m
[32m+[m[32m    #             downsample_method  = 'zero_pad' # 'variable', 'zero_pad', 'zero_pad_sym'[m
[32m+[m[32m    #             # NOTE: Variable length zero padding is logically incorrect,[m
[32m+[m[32m    #             # Because it means varible low-pass filtering, high for low-durated audios, and low for high-duration adious[m
[32m+[m[32m    # )[m
[32m+[m[41m    [m
     classification_layer = NNmodel([m
[31m-        NNtype= 'Linear', # 'Conv', 'FC', 'Linear'[m
[32m+[m[32m        NNtype= 'FC', # 'Conv', 'FC', 'Linear'[m
         down_sample_no= down_sample_no,[m
         hidden_layer_size = hidden_layer_size,[m
         num_classes= 10,[m
[31m-        dropout_prob= 0.1,[m
[32m+[m[32m        dropout_prob= 0.5,[m
     )[m
 [m
     print("Number of learnable parameters are: ", sum(p.numel() for p in classification_layer.parameters()))[m
 [m
[32m+[m[32m    # kfold_cross_validation([m
[32m+[m[32m    #     model           = classification_layer,[m
[32m+[m[32m    #     num_epoch       = num_epoch,[m
[32m+[m[32m    #     dataset         = dataset,[m
[32m+[m[32m    #     num_projections = num_projections,[m
[32m+[m[32m    #     batch_size      = batch_size,[m
[32m+[m[32m    #     k_folds         = 5[m[41m      [m
[32m+[m[32m    # )[m
[32m+[m
[32m+[m[32m    train_and_test([m
[32m+[m[32m        model= classification_layer,[m
[32m+[m[32m        num_epoch= num_epoch,[m
[32m+[m[32m        dataset= dataset,[m
[32m+[m[32m        batch_size= batch_size[m
[32m+[m[32m    )[m
 [m
[31m-    kfold_cross_validation([m
[31m-        model           = classification_layer,[m
[31m-        num_epoch       = num_epoch,[m
[31m-        dataset         = dataset,[m
[31m-        num_projections = num_projections,[m
[31m-        batch_size      = batch_size,[m
[31m-        k_folds         = 10      [m
[32m+[m[32m    classification_layer.load_state_dict(torch.load("saved_model.pt"))[m
[32m+[m[32m    model = classification_layer.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))[m
[32m+[m[32m    model.eval()[m
[32m+[m[32m    test_dataset = np.load("test_set.npy", allow_pickle=True)[m
[32m+[m[32m    test([m
[32m+[m[32m        model,[m
[32m+[m[32m        test_dataset,[m
[32m+[m[32m        device=  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')[m
     )[m
[32m+[m
[32m+[m[32m    # from librosa import display[m
[32m+[m[32m    # fig, ax = plt.subplots()[m
[32m+[m[32m    # S = np.abs(librosa.stft(self.dataset_list[0][:-1], n_fft=256))[m
[32m+[m[32m    # img = librosa.display.specshow(librosa.amplitude_to_db(S,[m
[32m+[m[32m    #                                                     ref=np.max),[m
[32m+[m[32m    #                             y_axis='linear', x_axis='time', ax=ax)[m
[32m+[m[32m    # ax.set_title('Power spectrogram')[m
[32m+[m[32m    # fig.colorbar(img, ax=ax, format="%+2.0f dB")[m
\ No newline at end of file[m
