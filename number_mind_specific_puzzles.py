__author__ = 'raphey'

# Some specific puzzles to play with

from number_mind import NumberMindPuzzle

demo_puzzle = NumberMindPuzzle(guesses=['90342', '70794', '39458', '34109', '51545', '12531'],
                               scores=[2, 0, 2, 1, 2, 1], solution='39542')

demo_puzzle_impossible = NumberMindPuzzle(guesses=['90342', '70794', '39458', '34109', '51545', '12531', '39542'],
                                          scores=[2, 0, 2, 1, 2, 1, 0])

standard_puzzle = NumberMindPuzzle(guesses=['5616185650518293', '3847439647293047', '5855462940810587',
                                            '9742855507068353', '4296849643607543', '3174248439465858',
                                            '4513559094146117', '7890971548908067', '8157356344118483',
                                            '2615250744386899', '8690095851526254', '6375711915077050',
                                            '6913859173121360', '6442889055042768', '2321386104303845',
                                            '2326509471271448', '5251583379644322', '1748270476758276',
                                            '4895722652190306', '3041631117224635', '1841236454324589',
                                            '2659862637316867'],
                                   scores=[2, 1, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 1, 2, 0, 2, 2, 3, 1, 3, 3, 2],
                                   solution='4640261571849533')

hard_puzzle1 = NumberMindPuzzle(guesses=['16850864592324350', '87918658310818734', '17857024007051755',
                                         '61396679548639575', '19501367190229270', '56887483185941298',
                                         '32961904128518336', '44874090871787141', '90108311014373109',
                                         '80405107988195249', '74513622219557530', '83066320528217707',
                                         '23016561002833870', '69832417123325318', '53414698784245529',
                                         '23140326224249719', '12868419232639415', '86808659510131676',
                                         '89688874695046174', '11934921862404923', '13247117491150159'],
                                scores=[2, 3, 2, 3, 1, 2, 1, 3, 2, 3, 3, 1, 2, 3, 0, 2, 1, 1, 3, 1, 3],
                                solution='64890127945863134')


hard_puzzle2 = NumberMindPuzzle(guesses=['006714750004835224', '829338081178445683', '740650266276793820',
                                         '741686590585691293', '233831062927676682', '963462544791983981',
                                         '789917402220608259', '868783999980896077', '408260980712457967',
                                         '090009451002198403', '926778364218013131', '052539719090479358',
                                         '253108549973809494', '327858986297168828', '432308210635627948',
                                         '409315678470111176', '624318788949746273', '625867266301070650',
                                         '245739857317374879', '701974634159899106', '133691634173898446',
                                         '287600865423475414', '550530265198327031', '653639977307243794',
                                         '566282104492705807', '928605715864736697', '645959108745559813',
                                         '248933509999057768', '936384741170157886'],
                                scores=[3, 2, 2, 1, 2, 2, 2, 1, 1, 2, 2, 3, 3, 3, 1,
                                        1, 1, 1, 3, 1, 0, 1, 0, 3, 1, 2, 3, 1, 2],
                                solution='942755483507209384')

generated_puzzle_length8 = NumberMindPuzzle(guesses=['63955935', '62823077', '30656661', '63918276', '78754401',
                                                     '71931071', '51872144', '02055748', '91366259', '13850861',
                                                     '48753963', '01766686', '79859027', '41231811'],
                                            scores=[1, 0, 2, 1, 3, 1, 1, 1, 3, 1, 2, 3, 1, 2])

generated_puzzle_length10 = NumberMindPuzzle(guesses=['1274907331', '7861576016', '3389625019', '8657282896',
                                                      '9439546289', '4601889829', '8271837850', '0710964729',
                                                      '7223830712', '2160342860', '9086105385', '8815227430',
                                                      '5134856041', '9205312516', '6539383155', '1443839428'],
                                             scores=[1, 0, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 2, 3, 2, 1],
                                             solution='9534232599')

generated_puzzle_length12 = NumberMindPuzzle(guesses=['755214215283', '651422950188', '739646920738', '483408970908',
                                                      '481052271548', '328288665274', '497206245855', '078556835952',
                                                      '279888234614', '310308953665', '032488064247', '987219395661',
                                                      '436289232222', '765817428137', '761363530099', '244637480673',
                                                      '951088298284', '900013116089', '526362154019', '901978433932',
                                                      '598213818406', '708107707295', '454878188646', '162657527857'],
                                             scores=[2, 2, 0, 1, 1, 2, 2, 1, 3, 2, 2, 2, 1, 2, 1, 3, 1, 1, 2, 1, 1, 1,
                                                     2, 1], solution='864482415615')

generated_puzzle_length14 = NumberMindPuzzle(guesses=['755214215283', '651422950188', '739646920738', '483408970908',
                                                      '481052271548', '328288665274', '497206245855', '078556835952',
                                                      '279888234614', '310308953665', '032488064247', '987219395661',
                                                      '436289232222', '765817428137', '761363530099', '244637480673',
                                                      '951088298284', '900013116089', '526362154019', '901978433932',
                                                      '598213818406', '708107707295', '454878188646', '162657527857'],
                                             scores=[2, 2, 0, 1, 1, 2, 2, 1, 3, 2, 2, 2, 1, 2, 1, 3, 1, 1, 2, 1, 1, 1,
                                                     2, 1], solution='864482415615')

generated_puzzle_length16 = NumberMindPuzzle(guesses=['755214215283', '651422950188', '739646920738', '483408970908',
                                                      '481052271548', '328288665274', '497206245855', '078556835952',
                                                      '279888234614', '310308953665', '032488064247', '987219395661',
                                                      '436289232222', '765817428137', '761363530099', '244637480673',
                                                      '951088298284', '900013116089', '526362154019', '901978433932',
                                                      '598213818406', '708107707295', '454878188646', '162657527857'],
                                             scores=[2, 2, 0, 1, 1, 2, 2, 1, 3, 2, 2, 2, 1, 2, 1, 3, 1, 1, 2, 1, 1, 1,
                                                     2, 1], solution='864482415615')

generated_puzzle_length18 = NumberMindPuzzle(guesses=['755214215283', '651422950188', '739646920738', '483408970908',
                                                      '481052271548', '328288665274', '497206245855', '078556835952',
                                                      '279888234614', '310308953665', '032488064247', '987219395661',
                                                      '436289232222', '765817428137', '761363530099', '244637480673',
                                                      '951088298284', '900013116089', '526362154019', '901978433932',
                                                      '598213818406', '708107707295', '454878188646', '162657527857'],
                                             scores=[2, 2, 0, 1, 1, 2, 2, 1, 3, 2, 2, 2, 1, 2, 1, 3, 1, 1, 2, 1, 1, 1,
                                                     2, 1], solution='864482415615')

generated_puzzle_length20 = NumberMindPuzzle(guesses=['75040488990180251202', '67419366357242237950',
                                                      '09865561751973353519', '79694249612326785533',
                                                      '53468697339656280595', '66626736442371819565',
                                                      '65777057250007209009', '66187039015252346347',
                                                      '76088625264875275090', '41555364668199942813',
                                                      '03597933154111219098', '27239852916293266571',
                                                      '04838716905616066087', '98247163536088651984',
                                                      '01056044878319767492', '56204136650587968127',
                                                      '58953859360731001941', '11018296056884437619',
                                                      '61670693151354459372', '78178284991528447999',
                                                      '02931631278575898398', '22227949264531254835',
                                                      '43693146150028441301', '15762214497649541830',
                                                      '18868184152357742086', '34514018502073697473',
                                                      '62777708613613739461', '05596473420714155389',
                                                      '69332955918142831403', '55685386406025321767',
                                                      '42822950807004024284', '15015824328512616038',
                                                      '54130878448088545749', '67121849833770290788',
                                                      '28683823423563978802', '28029894681534055210'],
                                             scores=[0, 2, 3, 2, 2, 1, 2, 2, 2, 1, 3, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2,
                                                     1, 1, 1, 1, 1, 3, 3, 1, 3, 3, 1, 2, 2, 2],
                                             solution='36799995728966304988')

