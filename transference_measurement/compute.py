import pandas
import plotly.express as px
import numpy
from plotly_utils import set_boring_thesis_template_as_default

set_boring_thesis_template_as_default()

PLOTS_SPLITS = dict(
	color = 'board',
	facet_row = 'type',
)

def mean_sample_wise(x):
	return numpy.array(numpy.stack((x[:-1],x[1:]))).mean(axis=0)

def compute_fft(t, a):
	f = numpy.fft.rfftfreq(
		n = len(t),
		d = numpy.diff(t)[0],
	)
	fft = numpy.fft.rfft(a)
	retval = pandas.DataFrame({'Frequency (Hz)': f, 'FFT': fft})
	return retval

input_chubut = pandas.read_csv('measured_data/Chubut_2/input.csv')
output_chubut = pandas.read_csv('measured_data/Chubut_2/output_more.csv')
output_santa = pandas.read_csv('measured_data/Santa_Cruz/output.csv')

for _ in [input_chubut,output_chubut,output_santa]:
	_.rename(columns={'Time': 'Time (s)', 'Ampl': 'Amplitude (V)'}, inplace=True)

output_chubut['Amplitude (V)'] *= -1
output_chubut['Amplitude (V)'] /= output_chubut['Amplitude (V)'].max()
output_santa['Amplitude (V)'] /= output_santa['Amplitude (V)'].max()
input_chubut['Amplitude (V)'] *= -1
input_chubut['Amplitude (V)'] /= input_chubut['Amplitude (V)'].max()

input_chubut['board'] = 'Chubut 2'
output_chubut['board'] = 'Chubut 2'
output_santa['board'] = 'Santa Cruz'
input_chubut['type'] = 'input'
output_chubut['type'] = 'output'
output_santa['type'] = 'output'

original_data = pandas.concat([output_chubut,output_santa,input_chubut])
original_data.set_index(['type','board'], inplace=True)
original_data.rename(columns={'Amplitude (V)': 'Normalized amplitude'}, inplace=True)

fig = px.line(
	title = 'Measured signals',
	data_frame = original_data.reset_index(drop=False).sort_values(['board','type','Time (s)']),
	x = 'Time (s)',
	y = 'Normalized amplitude',
	**PLOTS_SPLITS,
)
fig.write_html(
	'measured_waveforms.html',
	include_plotlyjs = 'cdn',
)

impulse_data = original_data['Normalized amplitude'].groupby(['type','board']).apply(numpy.diff)
time_for_imulse_data = original_data['Time (s)'].groupby(['type','board']).apply(mean_sample_wise)
impulse_data = pandas.concat([impulse_data,time_for_imulse_data], axis=1)

impulse_data = impulse_data.explode(list(impulse_data.columns))

fig = px.line(
	title = '<sup>d(measured signals)</sup>/<sub>d(time)</sup>',
	data_frame = impulse_data.reset_index(drop=False).sort_values(['board','type','Time (s)']),
	x = 'Time (s)',
	y = 'Normalized amplitude',
	**PLOTS_SPLITS,
)
fig.write_html(
	'impulse_waveforms.html',
	include_plotlyjs = 'cdn',
)

fft = impulse_data.groupby(['type','board']).apply(lambda x: compute_fft(x['Time (s)'],x['Normalized amplitude']))
fft.index = fft.index.droplevel(2)
fft.loc[(fft.index.get_level_values('board')=='Chubut 2')&(fft.index.get_level_values('type')=='output'),'FFT'] /= 1.5
fft['Frequency (GHz)'] = fft['Frequency (Hz)']*1e-9
fft['abs(FFT)'] = abs(fft['FFT'])
fft['abs(FFT) (dB)'] = 20*numpy.log10(fft['abs(FFT)'])

fig = px.line(
	title = 'FFT',
	data_frame = fft.reset_index(drop=False),
	x = 'Frequency (GHz)',
	y = 'abs(FFT) (dB)',
	log_x = True,
	# ~ log_y = True,
	**PLOTS_SPLITS,
)
fig.add_hline(
	y = -3,
	line_dash = "dash",
	annotation_text = '-3 dB',
	line_width = 2,
)
fig.update_xaxes(range=[numpy.log10(.01),numpy.log10(1)])
fig.update_yaxes(range=[-5,1])
fig.write_html(
	'fft.html',
	include_plotlyjs = 'cdn',
)
