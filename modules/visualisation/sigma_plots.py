import math
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy.stats

from matplotlib.ticker import FormatStrFormatter


def confidence_interval(p, std, n):
    return scipy.stats.t.ppf(p, n) * (std / math.sqrt(n))


def plot(n, ci_p, data, to_plot):
    with plt.style.context('seaborn-whitegrid'):
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['mathtext.fontset'] = 'dejavuserif'

        if to_plot == "f1":
            fig, ax = plt.subplots()
            sigmas = data['X'][:-6]
            for i, (means, stds, label) in enumerate(data['Y']):
                cis = [confidence_interval(ci_p, std, n) for std in stds[:-6]]
                means = [round(x,3) for x in means[:-6]]
                ax.errorbar(sigmas, means, yerr=cis, fmt=f'C{i}.-', ecolor=f'C{i}', elinewidth=1, capsize=1, linewidth=2, label=label)
            
            entropy_sigmas = [0.0656, 0.0684, 0.0701, 0.0716]
            for i, (means, _, _) in enumerate(data['Y']):
                means = [round(x,3) for x in means[:-6]]

                # plot horizontal lines
                max_f1 = np.max(means)
                ax.axhline(y=max_f1, xmin=0, xmax=1, color=f'C{i}', linewidth=1)
                trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
                ax.text(0, max_f1, f'{round(max_f1,2)}', color=f'C{i}', transform=trans, ha="right", va="center")

                # plot vertical lines
                max_sigma = sigmas[np.argmax(means)]
                max_f1_axis = ax.transLimits.transform((max_sigma, max_f1))[1]
                ax.axvline(x=max_sigma, ymin=0, ymax=max_f1_axis, color=f'C{i}', linewidth=1, linestyle='--')
                trans = transforms.blended_transform_factory(ax.get_xticklabels()[0].get_transform(), ax.transData)
                ax.text(max_sigma+0.0003, 0.49, f'{round(max_sigma,3)}', color=f'C{i}', transform=trans, ha="left", va="top")

                # plot entropy heuristic sigma values
                ax.scatter(entropy_sigmas[i], 0.48, s=70, marker='x', color=f'C{i}', clip_on=False, zorder=10)
                # ax.text(entropy_sigmas[i], 0.48, 'H', color=f'C{i}', ha="center", va="center")
            
            ax.set_ylim(0.48,0.94)
            ax.set_ylabel(r'f1', fontsize='x-large', fontweight='medium', rotation='horizontal')
            ax.set_xlabel(r'$\sigma$', fontsize='x-large', fontweight='medium')
            ax.set_xticks([i for i in np.arange(0.04,0.13,0.01)])
            ax.tick_params(axis='x', pad=8)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
            ax.grid(b=True)
            plt.legend(loc=(0.9,0.25), frameon=True, fancybox=True, shadow=True, fontsize='x-large')

        elif to_plot=="aug_acc":
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            sigmas = data['X'][:-6]
            entropy_sigmas = [0.0656, 0.0684, 0.0701, 0.0716]
            optimal_sigmas = [0.04, 0.055, 0.075, 0.08]
            for i, (aa_means, aa_stds, label, fu_means, fu_stds) in enumerate(data['Y']):
                aa_cis = [confidence_interval(ci_p, std, n) for std in aa_stds[:-6]]
                fu_cis = [confidence_interval(ci_p, std, n) for std in fu_stds[:-6]]
                ax1.errorbar(sigmas, aa_means[:-6], yerr=aa_cis, fmt=f'C{i}.-', ecolor=f'C{i}', elinewidth=1, capsize=1, linewidth=2, label=label)
                ax2.errorbar(sigmas, fu_means[:-6], yerr=fu_cis, fmt=f'C{i}.-', ecolor=f'C{i}', elinewidth=1, capsize=1, linewidth=2, label=label)
                ax2.scatter(entropy_sigmas[i], 0.68, s=70, marker='x', color=f'C{i}', clip_on=False, zorder=10)
                ax2.scatter(optimal_sigmas[i], 0.68, s=70, marker='o', facecolors='none', edgecolors=f'C{i}', linewidths=2, clip_on=False, zorder=10)

            
            ax2.set_ylim(0.68,0.98)
            ax1.set_ylabel("augmentation accuracy", fontsize='large', fontweight='medium')
            ax2.set_ylabel("fraction of unlabeled data used", fontsize='large', fontweight='medium')
            ax2.set_xlabel(r'$\sigma$', fontsize='xx-large', fontweight='bold')
            ax2.tick_params(axis='x', pad=8)
            ax1.grid(b=True); ax2.grid(b=True)
            ax2.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize='x-large')
            fig.subplots_adjust(hspace=0.05)

        fig.set_size_inches(15, 10)
        plt.savefig(f'./ablation_study_{to_plot}.pdf', format='pdf', dpi=100)
        plt.show()


if __name__ == "__main__":
    
    f1_data = {
        'X': [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15],
        'Y': [
            (
                [0.9155504587155964, 0.915611620795107, 0.9156077981651376, 0.9157798165137615, 0.9157339449541284, 0.9158125819134993, 0.9157855504587157, 0.9157492354740062, 0.9157660550458717, 0.9158423686405337, 0.9159518348623853, 0.9159668313338037, 0.9159239842726081, 0.9158654434250766, 0.9157855504587157, 0.9156341068537508, 0.9155504587155964, 0.915456301303718, 0.9153876146788992, 0.9153101791175186, 0.9152418682235196, 0.915191463901077, 0.9151204128440369],
                [0.006853002219778642, 0.006838795056575913, 0.0068608178893579085, 0.0068629740217486445, 0.006880377156906687, 0.006880393800492305, 0.006832858578104147, 0.006850084643609216, 0.006868844467925583, 0.006995204955340093, 0.007023484184709567, 0.007076169712117489, 0.007178838516421521, 0.007256060424968686, 0.0073556278650454415, 0.007485629983276791, 0.007557152859583798, 0.007678542175544547, 0.00785801339433276, 0.007998623694265279, 0.008188755912174578, 0.008356927803399847, 0.008574100011590343],
                '0.9'
            ),
            (
                [0.8561467889908256, 0.8583180428134557, 0.8594151376146789, 0.8597889908256882, 0.8576146788990826, 0.8573197903014417, 0.8568635321100918, 0.8567023445463813, 0.8555963302752294, 0.8548999165971644, 0.8554396024464832, 0.8553316866619619, 0.8540989515072085, 0.8525810397553517, 0.8506422018348624, 0.848850512682137, 0.8464041794087666, 0.8439932399806858, 0.8416743119266055, 0.8396024464831805, 0.8369787322768973, 0.8348424411647387, 0.8315902140672783],
                [0.07765045697588169, 0.07387844397072671, 0.07443439672398551, 0.07331355112152858, 0.07584568891360571, 0.07643764563860256, 0.07739029501935507, 0.07813510100064805, 0.07931477860319178, 0.07989710616382241, 0.07895093319348101, 0.07898685368786094, 0.07994009261493132, 0.08087107041585984, 0.08194962794347554, 0.08289793560851956, 0.08419222761909542, 0.08560772750532476, 0.08751216143075757, 0.08859482692125796, 0.0903529853269403, 0.0917459092376802, 0.09409908673500979],
                '0.5'
            ),
            (
                [0.726376146788991, 0.7350764525993884, 0.73901376146789, 0.740908256880734, 0.7392813455657493, 0.7389449541284404, 0.7408256880733947, 0.7428338430173294, 0.7434311926605506, 0.7416430358632194, 0.7409288990825689, 0.7390049400141144, 0.736651376146789, 0.7346146788990824, 0.7331049311926605, 0.7303912574203992, 0.7278746177370031, 0.7248237566393047, 0.7212133027522937, 0.7171887287024903, 0.7132527105921601, 0.7093318707618668, 0.7056230886850153],
                [0.14061690736470875, 0.14013135735482485, 0.1409100078529022, 0.1402928745479801, 0.14006633588857592, 0.14035351553205613, 0.13949178500663162, 0.13797680146092717, 0.13780440865905402, 0.13776819879723848, 0.1365573802959919, 0.1363829959398227, 0.1359829351494853, 0.1357912852623975, 0.1351166793105484, 0.13497328833681188, 0.1351354927090111, 0.1350947351391811, 0.13570878998283234, 0.1361842407874692, 0.1367478166515124, 0.13766295146304824, 0.13849752484407435],
                '0.3'
            ),
            (
                [0.5424770642201835, 0.5459938837920489, 0.5455963302752294, 0.5461743119266055, 0.5475611620795108, 0.5476212319790301, 0.5480389908256881, 0.5483537206931702, 0.5487981651376147, 0.5482902418682236, 0.5483600917431193, 0.5482321806633733, 0.5477588466579293, 0.5464097859327217, 0.5437385321100918, 0.5416540744738262, 0.5396814475025484, 0.5372621921776919, 0.5339357798165137, 0.5308060288335518, 0.527550041701418, 0.5240167530913442, 0.5203899082568807],
                [0.13741823899952227, 0.13972384363671783, 0.13958651529759183, 0.14273617011135994, 0.14313624614208004, 0.1427800949837473, 0.14079481056985754, 0.13950690907881516, 0.13655932916855001, 0.13482026469564765, 0.13300450821820253, 0.13103017870305775, 0.12940002710085577, 0.12792027450540913, 0.12607660891849648, 0.12449171623083294, 0.12287006433924064, 0.12177209556405627, 0.12097014728420309, 0.12039524435771609, 0.11991163901830706, 0.11973566976210998, 0.11944472987040344],
                'OS'
            )
        ]
    }


    data_09 = [round(x,3) for x in f1_data['Y'][0][0]]
    data_05 = [round(x,3) for x in f1_data['Y'][1][0]]
    data_03 = [round(x,3) for x in f1_data['Y'][2][0]]
    data_OS = [round(x,3) for x in f1_data['Y'][3][0]]
    print((np.max(data_09) - 0.9158125819134993)/0.9158125819134993)
    print((np.max(data_05) - 0.8568635321100918)/0.8568635321100918)
    print((np.max(data_03) - 0.7408256880733947)/0.7408256880733947)
    print((np.max(data_OS) - 0.5480389908256881)/0.5480389908256881)
    print(np.max(data_09) - np.min(data_09))
    print(np.max(data_05) - np.min(data_05))
    print(np.max(data_03) - np.min(data_03))
    print(np.max(data_OS) - np.min(data_OS))
    print("")

    aug_acc_data = {
        'X': [0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09, 0.095, 0.1, 0.105, 0.11, 0.115, 0.12, 0.125, 0.13, 0.135, 0.14, 0.145, 0.15],
        'Y': [
            (
                [0.8542083333333333, 0.8627777777777778, 0.8699375, 0.87195, 0.8771111111111111, 0.8812380952380954, 0.88184375, 0.8852685185185186, 0.8911916666666667, 0.8963712121212121, 0.9002708333333334, 0.9039038461538462, 0.9065952380952382, 0.9089833333333333, 0.9113125, 0.9142549019607843, 0.9171574074074075, 0.9193728070175439, 0.9211416666666666, 0.9213174603174604, 0.9221818181818182, 0.9228478260869566, 0.9229375000000001],
                [0.13766782056618912, 0.13468985865569597, 0.13346372367873768, 0.13253145433778687, 0.13301067440687922, 0.1324193931854771, 0.13440340098765585, 0.13412308515643118, 0.13323557640301797, 0.13229458473821223, 0.13119305036517892, 0.1305790659479519, 0.1300128002594072, 0.12988436345600024, 0.1293405264725424, 0.12914525588070844, 0.12805130111301194, 0.128048447372769, 0.12952520963332023, 0.13456056153863338, 0.1371467446511633, 0.14307427223124763, 0.15002709872889383],
                '0.9',
                [0.9279166666666667, 0.9194444444444444, 0.9133333333333334, 0.9126666666666667, 0.905, 0.893690476190476, 0.8823958333333334, 0.8727777777777779, 0.8610833333333334, 0.8472727272727273, 0.834861111111111, 0.8215384615384616, 0.8111904761904761, 0.8003888888888888, 0.78765625, 0.7749019607843137, 0.7628240740740742, 0.751359649122807, 0.7377083333333334, 0.7240873015873016, 0.7116666666666667, 0.6963768115942028, 0.681423611111111],
                [0.1048071973249505, 0.10711566380366022, 0.11081766806586194, 0.11370331764924206, 0.11942377082070821, 0.12750611340996987, 0.1341349195746788, 0.13950861737178696, 0.14764613758728823, 0.15685558037437555, 0.16350466967696317, 0.171312756345002, 0.17617454882964453, 0.18119417829200138, 0.18926592448000468, 0.19583278730690276, 0.20220021226544757, 0.208232380896147, 0.21554587920363188, 0.2232586058293063, 0.22917341587857554, 0.23865345490814194, 0.24703531236166731],
            ),
            (
                [0.8212324022394675, 0.8257572973725147, 0.8323224473313576, 0.8378748969206063, 0.8421310403728326, 0.8469900090751659, 0.8516790807224902, 0.8567630800191248, 0.8620660806147558, 0.867993735573134, 0.8741728602001062, 0.8804833355836945, 0.8859915705080065, 0.8910753622164584, 0.8952388887171127, 0.8989638401335881, 0.9029314409709726, 0.9066519716469752, 0.9101302718268119, 0.9135305582168638, 0.9166892256909266, 0.9194534558001914, 0.9223716935425235],
                [0.07893195230154683, 0.0758068591662187, 0.07367867987018502, 0.07368753015222518, 0.07450453070932989, 0.0743891202006972, 0.07443228568087189, 0.07433682980951309, 0.07492729839718823, 0.07594844085132445, 0.0768385929727415, 0.07799738834433578, 0.07865370762375684, 0.07913140418065713, 0.0792651626183828, 0.0789731558405205, 0.07905619714417646, 0.07916034555707026, 0.07911552648260076, 0.0790665377556412, 0.07897191318119628, 0.07874781599147054, 0.07861928487551545],
                '0.5',
                [0.9312962962962962, 0.9253086419753087, 0.9167129629629629, 0.9067037037037037, 0.8964506172839507, 0.8871693121693122, 0.876273148148148, 0.8656995884773662, 0.8546111111111111, 0.8414141414141414, 0.8299691358024692, 0.8178347578347579, 0.8050132275132275, 0.7922098765432098, 0.7802314814814815, 0.767875816993464, 0.7545679012345679, 0.7407602339181287, 0.7263518518518518, 0.7120546737213403, 0.6979040404040404, 0.6834541062801932, 0.6684104938271606],
                [0.048346812367703926, 0.050643624582638685, 0.054801968433351875, 0.05925508087120883, 0.0650083456942801, 0.06950161415980413, 0.07630528182772117, 0.0825656121286967, 0.08911198661864791, 0.09816013117647648, 0.10407286128344513, 0.111165024541487, 0.11891241514011396, 0.12660217776935231, 0.13300170145213644, 0.1396500221704007, 0.14807127635250003, 0.15689113701615368, 0.16647753775248986, 0.1756816033398133, 0.18423079730742523, 0.19330503606358101, 0.20316663147755865],
            ),
            (
                [0.7652354652273223, 0.7720423232826772, 0.7786279929728229, 0.7841076443978495, 0.789615853549436, 0.794806239830614, 0.8006974124395491, 0.8074415113972496, 0.8139636559312402, 0.8200869194715442, 0.82800872187045, 0.8349359639501474, 0.8420658801956861, 0.8487247463059022, 0.8555320448053564, 0.8615569169767191, 0.8670239194929359, 0.8721654033721739, 0.8772550269503578, 0.8822317230684972, 0.8867180843601422, 0.8907708333992358, 0.8946307017631306],
                [0.09547942339223434, 0.09332961082971483, 0.09413372632094309, 0.09294039266195302, 0.09285531352455108, 0.09340584549613663, 0.09404995948808417, 0.09389439807798612, 0.09437503898963479, 0.09474755505446653, 0.09603033107576925, 0.09700464443125552, 0.09807413086778935, 0.09900996202008781, 0.09996250898556329, 0.10058553819423988, 0.1008954281712705, 0.10109313854062293, 0.10138469899291735, 0.1016868700528397, 0.1017166034510262, 0.1016060696599038, 0.10145100157796108],
                '0.3',
                [0.9275, 0.9226315789473684, 0.9147697368421052, 0.9032105263157894, 0.8905482456140351, 0.878063909774436, 0.8660361842105263, 0.8544005847953217, 0.8420921052631579, 0.8285167464114832, 0.8152302631578947, 0.800900809716599, 0.7864379699248119, 0.7730438596491228, 0.7594654605263158, 0.7455572755417957, 0.7312646198830409, 0.7165927977839336, 0.7019539473684211, 0.6876503759398496, 0.6732177033492823, 0.6583695652173913, 0.6441940789473684],
                [0.04356018379424455, 0.045578258574269524, 0.05028828551778793, 0.057247071076295286, 0.0657306718212061, 0.07265330850329284, 0.07911675735633396, 0.08472082707780643, 0.09179835992566289, 0.10089846148821299, 0.10830835919873522, 0.11773163405626681, 0.12679629359708716, 0.1340432597742993, 0.14165682142778988, 0.1494034463663983, 0.15793206575608537, 0.16697573826546583, 0.175699309010122, 0.18386794041410856, 0.19214139779148798, 0.20091680688550637, 0.2086629288434541],
            ),
            (
                [0.6318894814023728, 0.6372195283917474, 0.6432957303930619, 0.6501874185138975, 0.6557632806222087, 0.6621302309182818, 0.6700312702290685, 0.6772354362027343, 0.6855570355616197, 0.6931555934639386, 0.701692971677145, 0.7109013905175071, 0.7198653609584431, 0.7281207300917488, 0.7363685929837138, 0.7456757982671931, 0.7542397626476589, 0.7625710578864012, 0.7699790786486597, 0.7778395651718848, 0.7856319978924169, 0.7929712245906316, 0.8001796319779518],
                [0.11073888296624976, 0.11238873411394983, 0.1125428905662034, 0.11440127597776405, 0.1163269975973286, 0.11727738738453519, 0.1178853829432597, 0.11910438412501577, 0.12029023967848623, 0.12184050995019548, 0.12370695380861936, 0.1264609929630532, 0.1290964359413692, 0.13183321711029403, 0.13383851080103473, 0.13677738599899905, 0.13918363494748912, 0.14184418621771308, 0.14329216914009027, 0.1451244792135055, 0.1469745554123444, 0.14852651764009903, 0.14989124873618082],
                'OS',
                [0.9280208333333334, 0.9155902777777779, 0.9046614583333334, 0.8921041666666667, 0.8801041666666667, 0.8675744047619048, 0.8534895833333335, 0.8402777777777778, 0.8259895833333334, 0.8114299242424242, 0.7964149305555556, 0.7804967948717949, 0.765654761904762, 0.7508402777777777, 0.7354231770833334, 0.7202450980392155, 0.7053240740740742, 0.6903179824561404, 0.6749635416666667, 0.6599751984126985, 0.6452888257575757, 0.6301992753623188, 0.6158984375],
                [0.0410261973807929, 0.046889645345892524, 0.05229824799844253, 0.058567087452244794, 0.06466720025016116, 0.0711502498604274, 0.07985295663380113, 0.08810417515085392, 0.09710712277825367, 0.10607456913971454, 0.11547814245940574, 0.12540306379243918, 0.1336810675992196, 0.1420135253662829, 0.1510885342549491, 0.15996183028827032, 0.16805372218671424, 0.17640170501828042, 0.18518143662203576, 0.19337994669310832, 0.20120621365095182, 0.20963081007640866, 0.21672568497179512]
            )
        ]
    }

    to_plot = 'aug_acc'
    n = 200
    ci_p = 0.95
    plot(n, ci_p, aug_acc_data, to_plot)