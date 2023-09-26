clear all;
load centroidsList_HTNIAltPacman.mat  
px_to_mm = 0.0278; % from Arda

for j = 1:23
    curr_time_node_pos = centroidslist{j};
    HTNIaltpac_x_positions(:, j) = curr_time_node_pos(:, 1)*px_to_mm;
    HTNIaltpac_y_positions(:, j) = curr_time_node_pos(:, 2)*px_to_mm; 
end

top_3_leftmost = [19, 20, 21, 22, 23];

for j = 1:23

%     if ismember(j, top_3_leftmost)
%         [top_3_yvals,top_3_yindices] = mink(HTNIaltpac_y_positions(:, j), 3, 1);
%         
%         xvals_to_look_at = HTNIaltpac_x_positions(top_3_yindices, j);
%         min_xval = min(xvals_to_look_at);
% 
%         node_idx = find(HTNIaltpac_x_positions(:, j)==min_xval);
%         shift_idx(j) = node_idx;
%     
%     else
        [left_10_xvals,left_10_xindices] = mink(HTNIaltpac_x_positions(:, j), 10, 1);

        yvals_to_look_at = HTNIaltpac_y_positions(left_10_xindices, j);
        max_yval = max(yvals_to_look_at);

        node_idx = find(HTNIaltpac_y_positions(:, j)==max_yval);
        shift_idx(j) = node_idx;
%     end

end

for j = 1:23
    curr_shift_idx = shift_idx(j);
    
    HTNIaltpac_x_positions_transformed(:, j) = HTNIaltpac_x_positions(:,j) - HTNIaltpac_x_positions(curr_shift_idx,j);
    HTNIaltpac_y_positions_transformed(:, j) = HTNIaltpac_y_positions(:,j) - HTNIaltpac_y_positions(curr_shift_idx,j);

end

%% rotate matrices
theta = pi/2;
rot_matrix = [cos(theta) -sin(theta); sin(theta), cos(theta)];
for i = 1:23
    curr_pos(:, 1) = HTNIaltpac_x_positions_transformed(:, i);
    curr_pos(:, 2) = HTNIaltpac_y_positions_transformed(:, i);
    
    new_pos = rot_matrix*curr_pos';

    HTNIaltpac_x_pos_transformed(:, i) = new_pos(1, :);
    HTNIaltpac_y_pos_transformed(:, i) = new_pos(2, :);
end

for i = 1:23
    figure()
    plot(HTNIaltpac_x_positions(:, i), HTNIaltpac_y_positions(:, i), 'ro')
    hold on

    plot(HTNIaltpac_x_positions_transformed(:, i), HTNIaltpac_y_positions_transformed(:, i), 'ko')
    plot(HTNIaltpac_x_pos_transformed(:, i), HTNIaltpac_y_pos_transformed(:, i), 'bo')
    plot(0, 0, 'go')
end