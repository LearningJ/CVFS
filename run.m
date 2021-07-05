function [NMI_sets, max_NMI, max_ACC,best_v_para] = run(X, X_label,alphas, betas, num_feats, n_cluster, lambda, cluster_method, algo, block_size,dirname,data,incomplete_rate)
num_alpha = size(alphas, 2);
num_beta = size(betas, 2);
num_feat = size(num_feats, 2);
NMI_sets = cell(1, num_feat);
max_NMI = zeros(1 ,num_feat);
acc_sets = cell(1, num_feat);
max_ACC = zeros(1 ,num_feat);
num_view=size(X, 2);

max_acc=0;
max_nmi=0;
max_nmi_alpha=1;
max_nmi_beta=1;
max_acc_alpha=1;
max_acc_beta=1;
max_nmi_feat=1;
max_acc_feat=1;
for i=1:num_feat
    NMI_sets{i} =zeros(num_alpha, num_beta);
end
[ X_incomplete, info ] = construct_imcomplete_data( X, X_label, 1, incomplete_rate );
for i=1:num_alpha
    alpha = alphas(i);
    for k = 1:num_beta
        beta=betas(k);
        option = struct('pass', 1, 'buffer_size', block_size*2, 'beta', beta*ones(1, num_view), 'alpha', alpha*ones(1, num_view), ...
    'lambda',lambda*ones(1, num_view), 'maxIter', 200, 'num_cluster', n_cluster, 'block_size', block_size, 'gamma', 10^7 );
        if strcmp(algo, 'CVFS')==1
            [w, v, u] = CVFS(X, option);
        end
        for j=1:num_feat
            feat = num_feats(j);
            if strcmp(cluster_method, 'Kmeans')==1
                [NMI,indic]=run_Kmeans(feat, w, X_incomplete, info.label, n_cluster, algo);
            end
            if strcmp(cluster_method, 'MVC')==1
                [NMI,indic]= run_mvc(feat, w, X_incomplete, info.label, n_cluster);
            end
            result = bestMap(info.label, indic);
            acc = length(find(info.label == result'))/length(info.label);
            NMI_sets{j}(i, k) = NMI;
            acc_sets{j}(i, k) = acc;
            if NMI>max_nmi
                max_nmi=NMI;
                max_nmi_alpha=alpha;
                max_nmi_beta=beta;
                max_nmi_feat=feat;
                save([dirname,'\',data,'_',algo,'_nmi_w'], 'w');
            end
            if acc>max_acc
                max_acc=acc;
                max_acc_alpha=alpha;
                max_acc_beta=beta;
                max_acc_feat=feat;
                save([dirname,'\',data,'_',algo,'_acc_w'], 'w');
            end
         end
    
    end    
end


for i=1:num_feat
    max_NMI(i) = max(max(NMI_sets{i}));
    max_ACC(i) = max(max(acc_sets{i}));
end


best_v_para=struct('max_nmi', max_nmi, 'max_nmi_alpha',max_nmi_alpha, 'max_nmi_beta',max_nmi_beta,'max_nmi_feat',max_nmi_feat ...
    ,'max_acc', max_acc, 'max_acc_alpha',max_acc_alpha, 'max_acc_beta',max_acc_beta,'max_acc_feat',max_acc_feat);

end



