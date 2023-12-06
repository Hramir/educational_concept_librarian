import numpy as np
from utils.constant_utils import NUM_SBJS, NUM_DECADES, NUM_STATS, COLORS, CORTEX_TO_ABBREVIATION
from hyperbolic_clustering.hyperbolic_cluster_metrics import get_subnetwork_hyperbolic_radii_per_sbj_left_right
from visualization import get_cortices_and_cortex_ids_to_cortices
from visualization import get_average_roi_hyperbolic_radii_per_sbj_across_runs

NUM_ITERATIONS = 1000
def get_subnetwork_hyperbolic_radii_per_sbj_left_right_bootstrapped(date, precalculated_radii=None):
    """
    587 x 22 (NUM_SBJS x NUM_SUBNETS) Dictionary
    """
    subnetwork_hyperbolic_radii_per_sbj_L = [dict() for sbj_num in range(NUM_SBJS)]
    subnetwork_hyperbolic_radii_per_sbj_R = [dict() for sbj_num in range(NUM_SBJS)]
    if not precalculated_radii:
        radii_per_sbj_per_roi = get_average_roi_hyperbolic_radii_per_sbj_across_runs(date)
    else:
        radii_per_sbj_per_roi = precalculated_radii
        if type(radii_per_sbj_per_roi) != dict: raise AssertionError(f"Invalid precalculated_radii type! {type(precalculated_radii)}")
    for sbj_num in range(NUM_SBJS):
        for cortex_index in range(NUM_SUBNETS):
            subnetworks_L, subnetworks_R = get_subnetworks_left_right() # 0-indexed

            # TODO: CHECK THIS IS CORRECT
            subnetwork_radii_L = [radii_per_sbj_per_roi[sbj_num][index] for index in subnetworks_L[cortex_index]]
            subnetwork_radii_R = [radii_per_sbj_per_roi[sbj_num][index] for index in subnetworks_R[cortex_index]]
            # Make sure is by cortex index or by cortex string
            subnetwork_hyperbolic_radii_per_sbj_L[sbj_num][cortex_index] = sum(subnetwork_radii_L) / len(subnetwork_radii_L) 
            subnetwork_hyperbolic_radii_per_sbj_R[sbj_num][cortex_index] = sum(subnetwork_radii_R) / len(subnetwork_radii_R) 
    return subnetwork_hyperbolic_radii_per_sbj_L, subnetwork_hyperbolic_radii_per_sbj_R

def bootstrap(regression_type="linear"):
    """
    We have 587 subjects, evenly distributed among 7 decades. Each subject has 360 ROI hyperbolic radii, which are averaged into
    22 subnetwork radii according to each ROI's membership, and we average the 22 subnetwork radii of each subject according to 
    which decade each subject belongs to. 
    We end up with 22 vectors of size 7, where the statistic is plotted at each of the 7 decades. 
    We perform linear regression on the 22 statistics across the 7 decades, to obtain 1 slope per statistic, 
    so that we have 22 slopes. We want to boostrap the 587 subjects so that we can get 22 confidence intervals 
    about the different statistic slopes.
    """
    for _ in range(NUM_ITERATIONS):
        # Resample from the 587 subjects with replacement
        bootstrapped_subjects = np.random.choice(NUM_SBJS, NUM_SBJS, replace=True)
        # Get the 22 subnetwork radii for each of the 587 subjects
        subnetwork_hyperbolic_radii_per_sbj_L, subnetwork_hyperbolic_radii_per_sbj_R = get_subnetwork_hyperbolic_radii_per_sbj_left_right_bootstrapped(date, precalculated_radii=precalculated_radii)
        
        age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
        DECADES = [0, 30, 40, 50, 60, 70, 80, 90]
        subnets_to_decades_to_hyperbolic_radii = [dict() for _ in range(NUM_SUBNETS)]
        
        # rgb_colors = [mcolors.hex2color(COLORS[subnet_num]) for subnet_num in range(1, NUM_SUBNETS + 1)] # COLORS are 1 indexed due to HCP-MMP1 being 1-indexed
        _, cortex_ids_to_cortices = get_cortices_and_cortex_ids_to_cortices()
        rgb_colors = [mcolors.hex2color(COLORS[cortex_id]) for cortex_id in cortex_ids_to_cortices]

        # Get the 22 subnetwork radii for each of the 587 subjects, grouped by decade
        decade_sbj_num_lists_bootstrapped = [[sbj_num for sbj_num in bootstrapped_subjects \
            if DECADES[nth_index] <= age_labels[sbj_num] < DECADES[nth_index + 1]] for nth_index in range(NUM_DECADES - 1)]
        for subnet_num in range(NUM_SUBNETS):
            
            for nth_index, decade_sbj_num_list in enumerate(decade_sbj_num_lists_bootstrapped):
                subnet_radii = [(subnetwork_hyperbolic_radii_per_sbj_L[sbj_num][subnet_num] + \
                    subnetwork_hyperbolic_radii_per_sbj_R[sbj_num][subnet_num]) / 2 for sbj_num in decade_sbj_num_list]
                
                subnets_to_decades_to_hyperbolic_radii[subnet_num][nth_index] = subnet_radii
                
        # Get the 22 slopes for the 22 subnetworks across the 7 decades
        subnets_to_slopes = [dict() for _ in range(NUM_SUBNETS)]
        subnets_to_quadratic_coeffs = [dict() for _ in range(NUM_SUBNETS)]
        subnets_to_linear_coeffs = [dict() for _ in range(NUM_SUBNETS)]
        
        for subnet_num in range(NUM_SUBNETS):
            
            decades = np.arange(len(subnets_to_decades_to_hyperbolic_radii[subnet_num]))
            subnet_radius_across_decades = subnets_to_decades_to_hyperbolic_radii[subnet_num]
            x = decades
            y = subnet_radius_across_decades
            # subnets_to_slopes[subnet_num] = slope
            if regression_type == "linear":
                slope, intercept = np.polyfit(x, y, 1)
                subnets_to_slopes[subnet_num] = subnets_to_slopes.get(subnet_num, []) + [slope]
            elif regression_type == "quadratic":
                quadratic_coeff, linear_coeff, intercept = np.polyfit(x, y, 2)
                subnets_to_quadratic_coeffs[subnet_num] = subnets_to_quadratic_coeffs.get(subnet_num, []) + [quadratic_coeff]
                subnets_to_linear_coeffs[subnet_num] = subnets_to_linear_coeffs.get(subnet_num, []) + [linear_coeff]
            elif regression_type == "exponential":
                raise AssertionError("Not implemented yet!")    
            
    confidence_intervals = []
    confidence_intervals_quadratic = []
    # Create confidence intervals for the 22 slopes
    for subnet_num in range(NUM_SUBNETS):
        if regression_type == "linear":
            confidence_intervals.append(np.percentile(bootstrap_slopes[subnet_num], [2.5, 97.5], axis=0))
        elif regression_type == "quadratic":
            confidence_intervals_quadratic.append(np.percentile(bootstrap_quadratic_coeffs[subnet_num], [2.5, 97.5], axis=0))
            confidence_intervals.append(np.percentile(bootstrap_linear_coeffs[subnet_num], [2.5, 97.5], axis=0))
        elif regression_type == "exponential":
            raise AssertionError("Not implemented yet!")

    # Plot the 22 slopes with confidence intervals
    if regression_type == "linear":
        for i in range(len(confidence_intervals)):
            ax.barh(i, confidence_intervals[i][1] - confidence_intervals[i][0], left=confidence_intervals[i][0], height=0.8, color=COLORS[i + 1], alpha=0.7)    
            # ax.plot([confidence_intervals[i][0], confidence_intervals[i][1]], [i, i], color='black', linewidth=2)  # Whiskers
        ax.set_xlabel('Slope')
        subnetwork_names = [*CORTEX_TO_ABBREVIATION.keys()]
        ax.set_yticks(np.arange(len(confidence_intervals)))
        ax.set_yticklabels([subnetwork_names[i] for i in range(len(confidence_intervals))])
        ax.set_title('Confidence Intervals for 22 Subnetwork Coefficients')

        plt.show()
    elif regression_type == "quadratic":
        plt.figure()
        for i in range(len(confidence_intervals)):
            ax.barh(i, confidence_intervals[i][1] - confidence_intervals[i][0], left=confidence_intervals[i][0], height=0.8, color=COLORS[i + 1], alpha=0.7)    
            # ax.plot([confidence_intervals[i][0], confidence_intervals[i][1]], [i, i], color='black', linewidth=2)  # Whiskers
        ax.set_xlabel('Linear Coefficient')
        
        subnetwork_names = [*CORTEX_TO_ABBREVIATION.keys()]
        ax.set_yticks(np.arange(len(confidence_intervals)))
        ax.set_yticklabels([subnetwork_names[i] for i in range(len(confidence_intervals))])
        ax.set_title('Confidence Intervals for 22 Subnetwork Coefficients')

        # plt.show()
        
        plt.figure()
        for i in range(len(confidence_intervals)):
            ax.barh(i, confidence_intervals_quadratic[i][1] - confidence_intervals_quadratic[i][0], left=confidence_intervals_quadratic[i][0], height=0.8, color=COLORS[i + 1], alpha=0.7)    
            # ax.plot([confidence_intervals[i][0], confidence_intervals[i][1]], [i, i], color='black', linewidth=2)  # Whiskers
        ax.set_xlabel('Quadratic Coefficient')
        
        subnetwork_names = [*CORTEX_TO_ABBREVIATION.keys()]
        ax.set_yticks(np.arange(len(confidence_intervals)))
        ax.set_yticklabels([subnetwork_names[i] for i in range(len(confidence_intervals))])
        ax.set_title('Confidence Intervals for 22 Subnetwork Coefficients')

        plt.show()